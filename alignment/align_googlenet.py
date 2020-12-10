import os
import sys
import argparse
import torch
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models
import utils
from utils import alignment, data, attack
import definitions


parser = argparse.ArgumentParser(description='Aligns two GoogLeNets using cross-correlation of activations.')

parser.add_argument('--dir', type=str, default='model_dicts/paired_models/', metavar='DIR',
                    help='directory for saving model dicts (default: model_dicts/paired_models/)')
parser.add_argument('--dir2', type=str, default='model_data/paired_models/', metavar='DIR',
                    help='directory for saving paired model data (default: model_data/paired_models/)')
parser.add_argument('--data_path', type=str, default='data/', metavar='PATH',
                    help='path to datasets location (default: data/)')
parser.add_argument('--model_path', type=str, default='model_dicts/basic_models/', metavar='PATH',
                    help='path to models for pairing (default: model_dicts/basic_models/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: True)')
parser.add_argument('--transform', type=str, default='TinyTen', metavar='TRANSFORM',
                    help='transform name (default: TinyTen)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size (default: 512)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--epochs', type=int, default=200, metavar='EPOCHS',
                    help='Number of epochs the models were trained for')
parser.add_argument('--model', type=str, default='GoogLeNet', metavar='MODEL',
                    help='model name (default: GoogLeNet)')
parser.add_argument('--align_name', type=str, default='corr', metavar='ALIGN',
                    help='name for alignment type (default: corr)')
parser.add_argument('--adv_flag', type=bool, default=False, metavar='ALIGN',
                    help='adversarial flag (default:False)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--seed_a', type=int, default=None, metavar='S', help='seed init of model 0 (default: None)')
parser.add_argument('--seed_b', type=int, default=None, metavar='S', help='seed init of model 1 (default: None)')
args = parser.parse_args()

args.dir = ('%s%s/%s/' % (args.dir, args.model, args.dataset))
args.dir2 = ('%s%s/%s/' % (args.dir2, args.model, args.dataset))
args.model_path = ('%s%s/%s/' % (args.model_path, args.model, args.dataset))

project_root = definitions.get_project_root()
os.chdir(project_root)
os.makedirs(args.dir, exist_ok=True)
os.makedirs(args.dir2, exist_ok=True)

print('Arguments')
for arg in vars(args):
    print('%s: %s' % (arg, str(getattr(args, arg))))

model_paths = ['%scheckpoint_seed_%02d-%d.pt' % (args.model_path, args.seed_a, args.epochs),
               '%scheckpoint_seed_%02d-%d.pt' % (args.model_path, args.seed_b, args.epochs)]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test,
    shuffle_train=True,
    test_batch_size=args.batch_size
)

# Load the curve model
architecture = getattr(models, args.model)
checkpoint = [None] * 2
model = [None] * 2
for i in range(2):
    checkpoint[i] = torch.load(model_paths[i], map_location=device)
    model[i] = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
    model[i].load_state_dict(checkpoint[i]['model_state'])
    model[i].to(device)

if args.adv_flag:
    if args.dataset == 'TINY-IMAGENET-200':
        eps = 4.0 / 255
        eps_stp_sz = 1.0 / 255
    else:
        eps = 8.0 / 255
        eps_stp_sz = 2.0 / 255

    config = {
        'epsilon': eps,
        'num_steps': 5,
        'step_size': eps_stp_sz,
        'random_start': True,
        'loss_func': 'xent',
    }

    net0 = attack.AttackPGD(model[0], config, loss_func=utils.googlenet_criterion)
    net1 = attack.AttackPGD(model[1], config, loss_func=utils.googlenet_criterion)
    net0.to(device)
    net1.to(device)
else:
    net0 = None
    net1 = None

# Align the models
if args.model == 'GoogLeNet':
    align_obj = alignment.AlignedModelPairs(model[0], model[1], loaders['align'], adv_flag=args.adv_flag,
                                            net0=net0, net1=net1)
    print('Alignment object created.')
    align_obj.compute_moments()
    print('Moments computed')
    align_obj.compute_crosscorr()
    print('Cross-correlation matrix computed')
    align_obj.compute_match()
    print('Match computed')
    np.save('%smatch_%s_seeds_%02d_%02d.npy' % (args.dir, args.align_name, args.seed_a, args.seed_b), align_obj.matches)

print('Done')
