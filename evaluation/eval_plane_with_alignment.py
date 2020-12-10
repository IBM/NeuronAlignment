import argparse
import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models
from utils import utils, data, alignment
import definitions


parser = argparse.ArgumentParser(description='Evaluates the plane containing three loss optima. The third optima '
                                             'corresponds to the second optima aligned to the first optima.')

parser.add_argument('--dir', type=str, default='model_data/loss_planes_with_alignment/', metavar='DIR',
                    help='directory for saving planes (default: model_data/loss_planes_with_alignment/)')
parser.add_argument('--data_path', type=str, default='data/', metavar='PATH',
                    help='path to datasets location (default: data/)')
parser.add_argument('--model_path', type=str, default='model_dicts/basic_models/', metavar='PATH',
                    help='path to models for pairing (default: model_dicts/basic_models/)')
parser.add_argument('--alignment_path', type=str, default='model_dicts/paired_models/', metavar='PATH',
                    help='path to alignments (default: model_dicts/paired_models/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: True)')
parser.add_argument('--transform', type=str, default='TinyTen', metavar='TRANSFORM',
                    help='transform name (default: TinyTen)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size (default: 512)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--seed_a', type=int, default=None, metavar='S', help='random seed for model 0 (default: None)')
parser.add_argument('--seed_b', type=int, default=None, metavar='S', help='random seed for model 1(default: None)')

parser.add_argument('--num_points', type=int, default=17, metavar='N',
                    help='Number of points in each dimension (default: 17)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--epochs', type=int, default=250, metavar='EPOCHS',
                    help='Number of epochs the models were trained for')
parser.add_argument('--alignment_type', type=str, default=None,
                    help='the type of alignment to obtain the third optima')
args = parser.parse_args()

args.dir = ('%s%s/%s/' % (args.dir, args.model, args.dataset))
args.model_path = ('%s%s/%s/' % (args.model_path, args.model, args.dataset))
args.alignment_path = ('%s%s/%s/' % (args.alignment_path, args.model, args.dataset))

project_root = definitions.get_project_root()
os.chdir(project_root)
os.makedirs(args.dir, exist_ok=True)

print('Arguments')
for arg in vars(args):
    print('%s: %s' % (arg, str(getattr(args, arg))))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

print('device: %s' % str(device))

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)

model_dict0 = torch.load('%scheckpoint_seed_%02d-%d.pt' % (args.model_path, args.seed_a, args.epochs),
                         map_location=device)
model_dict1 = torch.load('%scheckpoint_seed_%02d-%d.pt' % (args.model_path, args.seed_b, args.epochs),
                         map_location=device)
matching = np.load('%smatch_%s_seeds_%02d_%02d.npy' %
                   (args.alignment_path, args.alignment_type, args.seed_a, args.seed_b), allow_pickle=True)

states = [None] * 3
states[0] = model_dict0['model_state']
states[1] = model_dict1['model_state']

architecture = getattr(models, args.model)
model_1 = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
model_1.load_state_dict(states[1])

if args.model == 'TinyTen' or args.model == 'VGG16':
    model_2 = alignment.align_models(model_1, matching)
elif args.model == 'ResNet32':
    model_2, matching = alignment.align_models_resnet(model_1, matching)
elif args.model == 'GoogLeNet':
    model_2 = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
    model_2.load_state_dict(states[1])
    model_2.align_inception(matching) 
else:
    raise ValueError('Not a valid neural network model name. Needs to be either TinyTen, VGG16, or ResNet32.')
model_2.to(device)
states[2] = model_2.state_dict()

googlenet_flag = args.model == 'GoogLeNet'
t1, t2, loss, acc, model1_coords, model2_coords = utils.eval_plane(states, architecture, loaders, num_classes,
                                                                   args.num_points, device, t_min=[-0.05, -0.05],
                                                                   t_max=[1.05, 1.05], googlenet_flag=googlenet_flag)

loss_plane = {'t1': t1, 't2': t2, 'loss': loss, 'acc': acc, 'model1_coords': model1_coords,
              'model2_coords': model2_coords}

np.save('%s%s_loss_plane_dict_seeds_%02d_%02d.npy' % (args.dir, args.alignment_type, args.seed_a, args.seed_b),
        loss_plane)
