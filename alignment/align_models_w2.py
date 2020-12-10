import os
import sys
import argparse
import torch
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models
from utils import alignment, data
import definitions


parser = argparse.ArgumentParser(description='Aligns two networks using the L2 distance between their activations.')

parser.add_argument('--dir', type=str, default='model_dicts/paired_models/', metavar='DIR',
                    help='directory for saving model dicts (default: model_dicts/paired_models/)')
parser.add_argument('--dir2', type=str, default='model_data/paired_models/', metavar='DIR',
                    help='directory for saving paired model data (default: model_data/paired_models/)')
parser.add_argument('--data_path', type=str, default='data/', metavar='PATH',
                    help='path to datasets location (default: data/)')
parser.add_argument('--model_path', type=str, default='model_dicts/basic_models/', metavar='PATH',
                    help='path to models for pairing (default: model_dicts/basic_models/) or adversarial_models')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: True)')
parser.add_argument('--transform', type=str, default='TinyTen', metavar='TRANSFORM',
                    help='transform name (default: TinyTen)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 512)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--epochs', type=int, default=200, metavar='EPOCHS',
                    help='Number of epochs the models were trained for')
parser.add_argument('--model', type=str, default='TinyTen', metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--align_name', type=str, default='w2_pre', metavar='ALIGN',
                    help='name for alignment type (default:w2_pre)')

parser.add_argument('--pre_act', action='store_false', default=True,
                    help='Indicates whether to align pre-activations instead of post-activations. (default: True)')

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
device = torch.device("cpu")

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

if args.dataset == 'TINY-IMAGENET-200':
    use_warmstart = False
else:
    use_warmstart = True
architecture = getattr(models, args.model)

checkpoint = [None] * 2
model = [None] * 2
for i in range(2):
    checkpoint[i] = torch.load(model_paths[i], map_location=device)
    model[i] = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
    model[i].load_state_dict(checkpoint[i]['model_state'])
    model[i].to(device)

if args.model == 'TinyTen' or args.model == 'VGG16' or args.model == 'ConvFC' or args.model == 'TinyThree':
    _, _, corr_unaligned_mean, _ = \
        alignment.compute_model_alignment_w2_pre(model[0], model[1], loaders['test_align'], pre_act=args.pre_act)
    print('Unaligned Mean Correlation', corr_unaligned_mean)
    hard_match, _, _, corr_aligned_train = \
        alignment.compute_model_alignment_w2_pre(model[0], model[1], loaders['align'], quad_assignment=False,
                                                 pre_act=args.pre_act)
    print('Aligned Mean Correlation', corr_aligned_train)
    hard_match = [i[:, 1] for i in hard_match]

    model[1] = alignment.align_models(model[1], hard_match)

    _, _, corr_aligned_mean, _ = \
        alignment.compute_model_alignment_w2_pre(model[0], model[1], loaders['test_align'], pre_act=args.pre_act)
    print('Aligned Mean Correlation', corr_aligned_mean)
    np.save('%smatch_%s_seeds_%02d_%02d.npy' % (args.dir, args.align_name, args.seed_a, args.seed_b), hard_match)

    np.save('%s%s_unaligned_mean_seeds_%02d_%02d.npy' % (args.dir2, args.align_name, args.seed_a, args.seed_b),
            corr_unaligned_mean)
    np.save('%s%s_aligned_mean_seeds_%02d_%02d.npy' % (args.dir2, args.align_name, args.seed_a, args.seed_b),
            corr_aligned_mean)
elif args.model == 'ResNet32':
    _, _, _, corr_unaligned_mean, _, _ = \
        alignment.compute_model_alignment_resnet(model[0], model[1], loaders['test_align'])

    hard_match, _, _, _, _, _ = \
        alignment.compute_model_alignment_resnet(model[0], model[1], loaders['align'])

    hard_match = [i[:, 1] for i in hard_match]

    model[1], _ = alignment.align_models_resnet(model[1], hard_match)

    _, _, _, corr_aligned_mean, _, _ = \
        alignment.compute_model_alignment_resnet(model[0], model[1], loaders['test_align'])

    np.save('%smatch_%s_seeds_%02d_%02d.npy' % (args.dir, args.align_name, args.seed_a, args.seed_b), hard_match)

    np.save('%s%s_unaligned_mean_seeds_%02d_%02d.npy' % (args.dir2, args.align_name, args.seed_a, args.seed_b),
            corr_unaligned_mean)
    np.save('%s%s_aligned_mean_seeds_%02d_%02d.npy' % (args.dir2, args.align_name, args.seed_a, args.seed_b),
            corr_aligned_mean)

else:
    raise ValueError('Not a valid neural network model name. Needs to be either TinyTen, VGG16, or ResNet32.')
