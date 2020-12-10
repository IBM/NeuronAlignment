import argparse
import os
import sys
import numpy as np
import torch
import copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models
from utils import utils, data, alignment
import definitions


parser = argparse.ArgumentParser(description='Evaluates the plane containing the quadratic Bezier curve between two '
                                             'models.')

parser.add_argument('--dir', type=str, default='model_data/loss_planes_with_curve/', metavar='DIR',
                    help='directory for saving planes (default: model_data/loss_planes_with_curve/)')
parser.add_argument('--dir_endpoints', type=str, default='model_dicts/basic_models/', metavar='ENDPOINTS',
                    help='directory to model dicts for the curve endpoints (default: model_dicts/basic_models/)')
parser.add_argument('--dir_midpoints', type=str, default='model_dicts/curve_models/same_seed/', metavar='MIDPOINTS',
                    help='directory to model dicts for the curve midpoints (default: model_dicts/curve_models/)')
parser.add_argument('--dir_alignment', type=str, default='model_dicts/paired_models/', metavar='DIR',
                    help='directory to alignments between the endpoint models (default: model_dicts/paired_models/)')

parser.add_argument('--num_points', type=int, default=17, metavar='N',
                    help='Number of points in each dimension (default: 17)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: True)')
parser.add_argument('--transform', type=str, default='TinyTen', metavar='TRANSFORM',
                    help='transform name (default: TinyTen)')
parser.add_argument('--data_path', type=str, default='data/', metavar='PATH',
                    help='path to datasets location (default: data/)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size (default: 512)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default='TinyTen', metavar='MODEL',
                    help='model name (default: TinyTen)')
parser.add_argument('--epochs_model',
                    type=int, default=200, metavar='EPOCHS', help='Number of epochs the models were trained for')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--seed_a', type=int, default=None, metavar='S', help='random seed for model 0 (default: None)')
parser.add_argument('--seed_b', type=int, default=None, metavar='S', help='random seed for model 1(default: None)')
parser.add_argument('--alignment', type=str, default='',
                    help='specify an alignment if the models are to be aligned before curve finding (default: None)')
parser.add_argument('--epochs_curve', type=int, default=200, metavar='S',
                    help='Number of epochs the curve was trained for')
args = parser.parse_args()

args.dir = ('%s%s/%s/' % (args.dir, args.model, args.dataset))
args.dir_endpoints = ('%s%s/%s/' % (args.dir_endpoints, args.model, args.dataset))
args.dir_midpoints = ('%s%s/%s/' % (args.dir_midpoints, args.model, args.dataset))
args.dir_alignment = ('%s%s/%s/' % (args.dir_alignment, args.model, args.dataset))

project_root = definitions.get_project_root()
os.chdir(project_root)
os.makedirs(args.dir, exist_ok=True)

if args.alignment == '':
    args.alignment = 'null'

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

model_dict0 = torch.load('%scheckpoint_seed_%02d-%d.pt' % (args.dir_endpoints, args.seed_a, args.epochs_model),
                         map_location=device)
model_dict1 = torch.load('%scheckpoint_seed_%02d-%d.pt' % (args.dir_endpoints, args.seed_b, args.epochs_model),
                         map_location=device)

states = [None] * 3
states[0] = model_dict0['model_state']
states[1] = model_dict1['model_state']

states[2] = torch.load('%scheckpoint_align_%s_seeds_%02d_%02d-%d.pt' %
                       (args.dir_midpoints, args.alignment, args.seed_a, args.seed_b, args.epochs_model),
                       map_location=device)
states[2] = states[2]['model_state']

architecture = getattr(models, args.model)
model_2 = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
model_2.load_state_dict(states[1])

if args.alignment != 'null':
    matching = np.load('%smatch_%s_seeds_%02d_%02d.npy' %
                       (args.dir_alignment, args.alignment, args.seed_a, args.seed_b), allow_pickle=True)
    if args.model == 'ResNet32':
        model_2, matching = alignment.align_models_resnet(model_2, matching)
    elif args.model == 'GoogLeNet': 
        model_2 = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
        model_2.load_state_dict(states[1])
        model_2.align_inception(matching)  
    else:
        model_2 = alignment.align_models(model_2, matching)
    model_2.to(device)
    states[1] = model_2.state_dict()

state_05 = copy.deepcopy(states[0])
t = 0.5
for i in states[0]:
    state_05[i] = (1-t)**2 * states[0][i] + 2*(1-t)*t * states[2][i] + t**2 * states[1][i]
states[2] = state_05

googlenet_flag = args.model == 'GoogLeNet'

t1, t2, loss, acc, model1_coords, model2_coords = utils.eval_plane(
    states, architecture, loaders, num_classes, args.num_points, device, t_min=[-0.05, -0.10], t_max=[1.05, 1.5],
    googlenet_flag=googlenet_flag)

loss_plane = {'t1': t1, 't2': t2, 'loss': loss, 'acc': acc, 'model1_coords': model1_coords,
              'model2_coords': model2_coords}

np.save('%s%s_loss_plane_dict_seeds_%02d_%02d-%d.npy' %
        (args.dir, args.alignment, args.seed_a, args.seed_b, args.epochs_curve), loss_plane)
