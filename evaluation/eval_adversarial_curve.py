import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models
from utils import utils, alignment, attack_curve, data
import definitions


parser = argparse.ArgumentParser(description='Evaluates an adversarially trained curve between two neural networks.')

parser.add_argument('--dir', type=str, default='model_data/adversarial_curves/', metavar='DIR',
                    help='directory for saving evaluated curve (default: model_data/curves/)')
parser.add_argument('--data_path', type=str, default='data/', metavar='PATH',
                    help='path to datasets location (default: data/)')
parser.add_argument('--dir_endpoints', type=str, default='model_dicts/adversarial_models/', metavar='ENDPOINTS',
                    help='directory to model dicts for the curve endpoints. (default: model_dicts/adversarial_models/)')
parser.add_argument('--dir_midpoints', type=str, default='model_dicts/adversarial_curve_models/same_seed/',
                    metavar='MIDPOINTS', help='directory to model dicts for the curve midpoints. '
                                              '(default: model_dicts/adversarial_curve_models/)')
parser.add_argument('--dir_alignment', type=str, default='model_dicts/adversarial_paired_models/', metavar='DIR',
                    help='directory to alignments between the endpoint models '
                         '(default: model_dicts/adversarial_paired_models/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: True)')
parser.add_argument('--transform', type=str, default='Adversarial', metavar='TRANSFORM',
                    help='transform name (default: Adversarial)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size (default: 512)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default='TinyTen', metavar='MODEL',
                    help='model name (default: TinyTen)')
parser.add_argument('--num_points', type=int, default=65, metavar='N',
                    help='number of points on the curve (default: 65)')
parser.add_argument('--epochs_model', type=int, default=200, metavar='EPOCHS',
                    help='Number of epochs the models was trained for')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: )')
parser.add_argument('--seed_a', type=int, default=None, metavar='S', help='random seed for model 0 (default: None)')
parser.add_argument('--seed_b', type=int, default=None, metavar='S', help='random seed for model 1(default: None)')
parser.add_argument('--epochs_curve', type=int, default=200,
                    help='The number of epochs the curve was trained for. Needed to access checkpoint.')
parser.add_argument('--alignment', type=str, default='',
                    help='specify an alignment if the models are to be aligned before curve finding (default: None)')
parser.add_argument('--eval_train', type=bool, default=False, help='Flag indicating to evaluate on the training set '
                                                                   'instead of test. (default: False)')
args = parser.parse_args()

args.dir = ('%s%s/%s/' % (args.dir, args.model, args.dataset))
args.dir_endpoints = ('%s%s/%s/' % (args.dir_endpoints, args.model, args.dataset))
args.dir_midpoints = ('%s%s/%s/' % (args.dir_midpoints, args.model, args.dataset))
args.dir_alignment = ('%s%s/%s/' % (args.dir_alignment, args.model, args.dataset))

project_root = definitions.get_project_root()
os.chdir(project_root)
os.makedirs(args.dir, exist_ok=True)

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
    args.use_test,
    test_batch_size=args.batch_size 
)

# Load the endpoint models
model_paths = ['%scheckpoint_seed_%02d-%d.pt' % (args.dir_endpoints, args.seed_a, args.epochs_model),
               '%scheckpoint_seed_%02d-%d.pt' % (args.dir_endpoints, args.seed_b, args.epochs_model)]
state_0 = torch.load(model_paths[0], map_location=device)
state_0 = state_0['model_state']
state_2 = torch.load(model_paths[1], map_location=device)
state_2 = state_2['model_state']

architecture = getattr(models, args.model)
model_0 = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
model_0.load_state_dict(state_0)
model_0.to(device)

model_2 = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
model_2.load_state_dict(state_2)

# Align the second endpoint model if necessary
if args.alignment is not None and args.alignment != '':
    matching = np.load('%smatch_%s_seeds_%02d_%02d.npy' %
                       (args.dir_alignment, args.alignment, args.seed_a, args.seed_b), allow_pickle=True)
    if args.model == 'ResNet32':
        model_2, _ = alignment.align_models_resnet(model_2, matching)
    elif args.model == 'GoogLeNet':
        model_2.align_inception(matching)
    else:
        model_2 = alignment.align_models(model_2, matching)
    model_2.to(device)
    state_2 = model_2.state_dict()
else:
    model_2.to(device)
    args.alignment = 'null'

# Load the midpoint model
state_1 = torch.load('%scheckpoint_align_%s_seeds_%02d_%02d-%d.pt' %
                     (args.dir_midpoints, args.alignment, args.seed_a, args.seed_b, args.epochs_curve),
                     map_location=device)
state_1 = state_1['model_state']
model_1 = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
model_1.load_state_dict(state_1)

model = architecture.curve(num_classes=num_classes, device=device, fix_points=[True]*3, **architecture.kwargs)
model.import_base_parameters(model_0, 0)
model.import_base_parameters(model_1, 1)
model.import_base_parameters(model_2, 2)
model.to(device)

# Parameters for adversarial training
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

if args.model == 'GoogLeNet':
    criterion = utils.googlenet_criterion
else:
    criterion = nn.CrossEntropyLoss()

net = attack_curve.AttackPGD(model, config, loss_func=criterion)

# Evaluate the curve with uniform sampling
t1 = np.linspace(0, 1.0, args.num_points)
has_bn = utils.check_bn(model)
loss = np.zeros([args.num_points])
acc = np.zeros([args.num_points])
loss_robust = np.zeros([args.num_points])
acc_robust = np.zeros([args.num_points])
t = torch.tensor([0, 0, 0], dtype=torch.float32, device=device, requires_grad=False)
model.eval()
model_0.eval()
model_1.eval()
model_2.eval()
for i, x in enumerate(t1):
    time_ep = time.time()
    t.copy_(torch.from_numpy(np.array([(1 - x) ** 2, 2 * (1-x) * x, x ** 2])))

    if has_bn:
        utils.update_bn(loaders['train'], net, t=t)
    if not args.eval_train:
        te_res = utils.test(loaders['test'], model, criterion, coeffs_t=t)
        te_res_robust = utils.test(loaders['test'], net, criterion, adversarial_flag=True, t=t)
    else:
        te_res = utils.test(loaders['train'], model, criterion, coeffs_t=t)
        te_res_robust = utils.test(loaders['train'], net, criterion, adversarial_flag=True, t=t)
    loss[i] = te_res['nll']
    acc[i] = te_res['accuracy']
    loss_robust[i] = te_res_robust['nll']
    acc_robust[i] = te_res_robust['accuracy']
    time_ep = time.time() - time_ep
    print('Point: %.3f, Test Loss: %.2E, Test Accuracy: %.2f, Robust Loss: %.2E, Robust Accuracy %.2f, Time Elapsed: '
          '%.2fs' % (x, te_res['nll'], te_res['accuracy'], te_res_robust['nll'], te_res_robust['accuracy'], time_ep))

# Store the coordinates of the endpoint models in the orthonormal plane that the curve lies on
_, _, _, _, model1_coord, model2_coord = \
    utils.create_model_ortho(state_0, state_2, state_1, architecture, num_classes, device, loaders['train'])
curve = np.zeros([args.num_points, 2])
if args.epochs_curve != 0:
    curve[:, 0] = 2 * (1-t1) * t1 * model2_coord[0] + t1 ** 2 * model1_coord[0]
    curve[:, 1] = 2 * (1-t1) * t1 * model2_coord[1] + t1 ** 2 * model1_coord[1]
else:
    curve[:, 0] = t1 * model1_coord[0]

# Save evaluation results
curve_len = np.cumsum(np.sum((curve[1:, :] - curve[:-1, :]) ** 2, axis=1) ** 0.5)
curve_len = np.concatenate([np.zeros(1), curve_len])

x = np.pad(curve_len, (1, 1), 'edge')
loss_obj = np.sum(((x[1:-1] - x[:-2])/2 + (x[2:] - x[1:-1])/2) * loss) / x[-1]
acc_obj = np.sum(((x[1:-1] - x[:-2])/2 + (x[2:] - x[1:-1])/2) * acc) / x[-1]

curve_eval = {'curve_len': curve_len, 'test_loss': loss, 'test_acc': acc, 'loss_line_integral': loss_obj,
              'acc_line_integral': acc_obj, 'curve_coords': curve, 'model1_coords': model1_coord,
              'model2_coords': model2_coord, 'test_loss_robust': loss_robust, 'test_acc_robust': acc_robust}

if not args.eval_train:
    np.save('%scurve_align_%s_seeds_%02d_%02d-%d.npy' % (args.dir, args.alignment, args.seed_a, args.seed_b,
                                                         args.epochs_curve), curve_eval)
else:
    np.save('%scurve_train_align_%s_seeds_%02d_%02d-%d.npy' % (args.dir, args.alignment, args.seed_a, args.seed_b,
                                                               args.epochs_curve), curve_eval)
