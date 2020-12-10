import os
import sys
import argparse
import torch
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import alignment, data
import models
from models import curves
import definitions


parser = argparse.ArgumentParser(description='Separately aligns the midpoint of a curve to both of its endpoints.')

parser.add_argument('--dir', type=str, default='model_dicts/paired_models_on_curve/', metavar='DIR',
                    help='directory for saving midpoint matching (default: model_dicts/paired_models_on_curve/)')
parser.add_argument('--dir2', type=str, default='model_data/paired_models_on_curve/', metavar='DIR',
                    help='directory for saving midpoint matching data (default: model_data/paired_models_on_curve/)')
parser.add_argument('--dir_endpoints', type=str, default='model_dicts/basic_models/', metavar='DIR',
                    help='directory to model dicts for the curve endpoints (default: model_dicts/basic_models/)')
parser.add_argument('--dir_midpoints', type=str, default='model_dicts/curve_models/', metavar='DIR',
                    help='directory to model dicts for the curve midpoints (default: model_dicts/curve_models/)')
parser.add_argument('--dir_alignment', type=str, default='model_dicts/paired_models/', metavar='DIR',
                    help='directory to alignments between the endpoint models (default: model_dicts/paired_models/)')

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

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name [TinyTen, ResNet32, GoogLeNet]')
parser.add_argument('--alignment', type=str, default=None, metavar='ALIGN',
                    help='Specify if an alignment was used when training the curve. (default: None), ['', corr]')
parser.add_argument('--point_t', type=int, default=0.5, metavar='N',
                    help='the point on the curve to align to each endpoint in the curve parameterization. '
                         '(default: 0.5)')
parser.add_argument('--curve_type', type=str, default='Bezier', metavar='CURVE',
                    help='parameterization of the curve (default: Bezier)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--seed_a', type=int, default=None, metavar='S', help='random seed for model 0')
parser.add_argument('--seed_b', type=int, default=None, metavar='S', help='random seed for model 1')

parser.add_argument('--epochs_model',
                    type=int, default=250, metavar='EPOCHS', help='Number of epochs the models were trained for')
parser.add_argument('--epochs_curve', type=int, default=250,
                    help='The number of epochs the curve was trained for. (default: 250)')
args = parser.parse_args()

args.dir = ('%s%s/%s/' % (args.dir, args.model, args.dataset))
args.dir2 = ('%s%s/%s/' % (args.dir2, args.model, args.dataset))
args.dir_endpoints = ('%s%s/%s/' % (args.dir_endpoints, args.model, args.dataset))
args.dir_midpoints = ('%s%s/%s/' % (args.dir_midpoints, args.model, args.dataset))
args.dir_alignment = ('%s%s/%s/' % (args.dir_alignment, args.model, args.dataset))

project_root = definitions.get_project_root()
os.chdir(project_root)
os.makedirs(args.dir, exist_ok=True)
os.makedirs(args.dir2, exist_ok=True)

print('Arguments')
for arg in vars(args):
    print('%s: %s' % (arg, str(getattr(args, arg))))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('device: %s' % str(device))

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)

# Load the endpoint models
model_paths = ['%scheckpoint_seed_%02d-%d.pt' % (args.dir_endpoints, args.seed_a, args.epochs_model),
               '%scheckpoint_seed_%02d-%d.pt' % (args.dir_endpoints, args.seed_b, args.epochs_model)]
state_0 = torch.load(model_paths[0], map_location=device)
state_0 = state_0['model_state']
state_2 = torch.load(model_paths[1], map_location=device)
state_2 = state_2['model_state']

num_bends = 3  # Currently, this version of the code only supports curves parameterized by 3 points
model_base = [None] * num_bends
architecture = getattr(models, args.model)
model_base[0] = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
model_base[0].load_state_dict(state_0)
model_base[0].to(device)

model_base[2] = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
model_base[2].load_state_dict(state_2)
model_base[2].to(device)

# Load the midpoint model with appropriate alignment
if args.alignment is not None and args.alignment != '':
    matching = np.load('%smatch_%s_seeds_%02d_%02d.npy' %
                       (args.dir_alignment, args.alignment, args.seed_a, args.seed_b), allow_pickle=True)
    if args.model == 'ResNet32':
        model_base[2], _ = alignment.align_models_resnet(model_base[2], matching)
    else:
        model_base[2] = alignment.align_models(model_base[2], matching)
    model_base[2].to(device)
    state_2 = model_base[2].state_dict()
else:
    args.alignment = 'null'

state_1 = torch.load('%scheckpoint_align_%s_seeds_%02d_%02d-%d.pt' %
                     (args.dir_midpoints, args.alignment, args.seed_a, args.seed_b, args.epochs_curve),
                     map_location=device)
state_1 = state_1['model_state']
model_base[1] = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
model_base[1].load_state_dict(state_1)

# Load the curve model
curve = getattr(curves, args.curve_type)
model = curves.CurveNet(
    num_classes,
    device,
    curve,
    architecture.curve,
    num_bends,
    architecture_kwargs=architecture.kwargs,
)

for i in range(num_bends):
    model.import_base_parameters(model_base[i], i)
model.to(device)

model_mid = model.compute_new_model(architecture.base, loaders['train'], args.point_t)
model_mid.to(device)

# Compute the correlation signature between the endpoints and the midpoint, before and after alignment
hard_match = [None] * 2
corr_unaligned_mean = [None] * 2
corr_aligned_mean = [None] * 2
if args.model == 'TinyTen' or args.model == 'VGG16':
    for i in range(2):
        hard_match[i], _, corr_unaligned_mean[i], corr_aligned_mean[i] = \
            alignment.compute_model_alignment(model_base[-i], model_mid, loaders['align'])

        hard_match[i] = [j[:, 1] for j in hard_match[i]]

    np.save('%smatch_corr_seeds_%02d_%02d_ends_align_%s-%d.npy' %
            (args.dir, args.seed_a, args.seed_b, args.alignment, args.epochs_curve), hard_match)
    np.save('%scorr_unaligned_mean_seeds_%02d_%02d_ends_align_%s-%d.npy' %
            (args.dir2, args.seed_a, args.seed_b, args.alignment, args.epochs_curve), corr_unaligned_mean)
    np.save('%scorr_aligned_mean_seeds_%02d_%02d_ends_align_%s-%d.npy' %
            (args.dir2, args.seed_a, args.seed_b, args.alignment, args.epochs_curve), corr_aligned_mean)
elif args.model == 'ResNet32':
    hard_match_noid = [None] * 2
    corr_aligned_noid_mean = [None] * 2
    for i in range(2):
        hard_match[i], hard_match_noid[i], _, corr_unaligned_mean[i], corr_aligned_mean[i], \
            corr_aligned_noid_mean[i] = \
            alignment.compute_model_alignment_resnet(model_base[-i], model_mid, loaders['align'])

        hard_match[i] = [j[:, 1] for j in hard_match[i]]
        hard_match_noid[i] = [j[:, 1] for j in hard_match_noid[i]]

    np.save('%smatch_corr_seeds_%02d_%02d_ends_align_%s-%d.npy' %
            (args.dir, args.seed_a, args.seed_b, args.alignment, args.epochs_curve), hard_match)
    np.save('%smatch_corr_noid_seeds_%02d_%02d_ends_align_%s-%d.npy' %
            (args.dir, args.seed_a, args.seed_b, args.alignment, args.epochs_curve), hard_match_noid)

    np.save('%scorr_unaligned_mean_seeds_%02d_%02d_ends_align_%s-%d.npy' %
            (args.dir2, args.seed_a, args.seed_b, args.alignment, args.epochs_curve), corr_unaligned_mean)
    np.save('%scorr_aligned_mean_seeds_%02d_%02d_ends_align_%s-%d.npy' %
            (args.dir2, args.seed_a, args.seed_b, args.alignment, args.epochs_curve), corr_aligned_mean)
    np.save('%scorr_aligned_noid_mean_seeds_%02d_%02d_ends_align_%s-%d.npy' %
            (args.dir2, args.seed_a, args.seed_b, args.alignment, args.epochs_curve), corr_aligned_noid_mean)
else:
    raise ValueError('Not a valid neural network model name. Needs to be either TinyTen, VGG16, or ResNet32.')
