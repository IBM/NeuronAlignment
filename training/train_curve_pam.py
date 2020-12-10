import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import curves_pam
from utils import utils, alignment, data
import models
import definitions
import copy


parser = argparse.ArgumentParser(description='Trains a curve between two neural networks using PAM.')

parser.add_argument('--dir', type=str, default='model_dicts/curve_pam_models/', metavar='DIR',
                    help='directory for saving curve models (default: model_dicts/curve_pam_models/)')
parser.add_argument('--dir2', type=str, default='model_data/training/curve_pam_models/', metavar='DIR',
                    help='directory for saving curve models data (default: model_data/training/curve_pam_models/)')
parser.add_argument('--data_path', type=str, default='data/', metavar='PATH',
                    help='path to datasets location (default: data/)')
parser.add_argument('--dir_models', type=str, default='model_dicts/basic_models/', metavar='ENDPOINTS',
                    help='directory to model dicts for the curve endpoints. (default: model_dicts/basic_models/)')
parser.add_argument('--dir_alignment', type=str, default='model_dicts/paired_models/', metavar='DIR',
                    help='directory to alignments between the endpoint models (default: model_dicts/paired_models/)')


parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: True)')
parser.add_argument('--transform', type=str, default='TinyTen', metavar='TRANSFORM',
                    help='transform name (default: TinyTen)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default='TinyTen', metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--curve', type=str, default='Bezier', metavar='CURVE',
                    help='curve type to use (default: Bezier)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')

parser.add_argument('--fix_start', dest='fix_start', action='store_true', default=True,
                    help='fix start point (default: True)')
parser.add_argument('--fix_end', dest='fix_end', action='store_true', default=True,
                    help='fix end point (default: True)')

parser.set_defaults(init_linear=True)
parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                    help='turns off linear initialization of intermediate points (default: on)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--outer_iters', type=int, default=1, metavar='N',
                    help='number of PAM iterations to train (default: 1)')
parser.add_argument('--inner_iters_perm', type=int, default=20, metavar='N',
                    help='number of epochs to train permutation for each subiteration. (default: 20)')
parser.add_argument('--inner_iters_phi', type=int, default=250, metavar='N',
                    help='number of epochs to train curve parameters for each subiteration. (default: 250)')

parser.add_argument('--save_freq', type=int, default=270, metavar='N',
                    help='save frequency (default: 270)')

parser.add_argument('--lr', type=float, default=1E-1, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--wd', type=float, default=5e-4, metavar='WD',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--lr_decay', type=float, default=0.9996, help='Learning Rate Decay for SGD')
parser.add_argument('--lr_drop', type=int, default=20, help='Number of epochs required to decay learning rate')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--seed_a', type=int, default=None, metavar='S', help='random seed for model 0 (default: None)')
parser.add_argument('--seed_b', type=int, default=None, metavar='S', help='random seed for model 1(default: None)')
parser.add_argument('--epochs_model',
                    type=int, default=200, metavar='EPOCHS', help='Number of epochs the models were trained for')
parser.add_argument('--alignment', type=str, default='',
                    help='specify an alignment if the models are to be aligned before curve finding (default: None)')
parser.add_argument('--val_freq', nargs='+', type=int, default=[20, 250],
                    help='the rate in epochs at which to evaluate the model on the validation set. (default: [20, 250])')

args = parser.parse_args()

args.dir = ('%s%s/%s/' % (args.dir, args.model, args.dataset))
args.dir2 = ('%s%s/%s/' % (args.dir2, args.model, args.dataset))
args.dir_models = ('%s%s/%s/' % (args.dir_models, args.model, args.dataset))
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

np.random.seed(args.seed)
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
    test_batch_size=512
)

model_paths = ['%scheckpoint_seed_%02d-%d.pt' % (args.dir_models, args.seed_a, args.epochs_model),
               '%scheckpoint_seed_%02d-%d.pt' % (args.dir_models, args.seed_b, args.epochs_model)]
state_0 = torch.load(model_paths[0], map_location=device)
state_1 = torch.load(model_paths[1], map_location=device)

architecture = getattr(models, args.model)
model_0 = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
model_0.load_state_dict(state_0['model_state'])
model_1 = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
model_1.load_state_dict(state_1['model_state'])

if args.alignment is not None and args.alignment != '' and args.alignment != 'pam':
    matching = np.load('%smatch_%s_seeds_%02d_%02d.npy' %
                       (args.dir_alignment, args.alignment, args.seed_a, args.seed_b), allow_pickle=True)
    if args.model == 'ResNet32':
        model_1, _ = alignment.align_models_resnet(model_1, matching)
    elif args.model == 'GoogLeNet':
        model_1.align_inception(matching)
    else:
        model_1 = alignment.align_models(model_1, matching)
    model_1.to(device)
else:
    matching = None

if args.model == 'GoogLeNet':
    matching_ref = np.load('%smatch_%s_seeds_%02d_%02d.npy' %
                       (args.dir_alignment, 'corr', args.seed_a, args.seed_b), allow_pickle=True)
else:
    matching_ref = None
curve = getattr(curves_pam, args.curve)
model = curves_pam.CurveNet(
    num_classes,
    device,
    curve,
    architecture.curve,
    args.num_bends,
    args.fix_start,
    args.fix_end,
    architecture_kwargs=architecture.kwargs,
    act_ref=matching_ref
)

perm_params = nn.ParameterList()
for param in model.permutations.parameters():
        if param.requires_grad:
            perm_params.append(param)

optimizer_perm = optim.SGD(
    perm_params,
    lr=(args.lr * 5E-1))

optimizer_phi = optim.SGD(
    filter(lambda param: param.requires_grad, model.curve_learnable_params),
    lr=args.lr,
    momentum=0.9,
    weight_decay=args.wd if args.curve is None else 0.0,
    nesterov=True)

lambda_perm = lambda epoch: 0.5 ** (epoch // 20) * args.lr_decay ** epoch
lambda_phi = lambda epoch: 0.5 ** (epoch // args.lr_drop) * args.lr_decay ** epoch

scheduler_perm = optim.lr_scheduler.LambdaLR(optimizer_perm, lr_lambda=lambda_perm)
scheduler_phi = optim.lr_scheduler.LambdaLR(optimizer_phi, lr_lambda=lambda_phi)

if args.resume is None:
    model.import_base_parameters(model_0, 0)
    model.import_base_parameters(model_1, 2)
    if args.init_linear:
        print('Linear initialization.')
        model.init_zero()
    start_epoch = 1
model.to(device)

model_turningpt = architecture.base(num_classes=num_classes, device=device)
model.export_base_parameters(model_turningpt, 1)

if args.model == 'GoogLeNet':
    criterion = utils.googlenet_criterion
else:
    criterion = nn.CrossEntropyLoss()
regularizer = None if args.curve is None else curves_pam.l2_regularizer(args.wd)

if args.val_freq is None:
    args.val_freq = np.nan

total_iters = args.outer_iters * (args.inner_iters_perm + args.inner_iters_phi)
acc_train = np.ones(total_iters + 1) * np.nan
acc_test = np.ones(total_iters + 1) * np.nan
loss_train = np.ones(total_iters + 1) * np.nan
loss_test = np.ones(total_iters + 1) * np.nan
has_bn = utils.check_bn(model)
lr = args.lr

change_P = np.ones(total_iters + 1) * np.nan

number_batches = len(loaders['test'])
loss_time = np.ones([total_iters+1, number_batches]) * np.nan
acc_time = np.ones([total_iters+1, number_batches]) * np.nan

if args.val_freq[0] is None:
    args.val_freq[0] = np.nan
if args.val_freq[1] is None:
    args.val_freq[1] = np.nan

print('Beginning training')
for iter in range(start_epoch, args.outer_iters + 1):

    params_before = [None] * len(optimizer_perm.param_groups[0]['params'])
    for idx, param in enumerate(optimizer_perm.param_groups[0]['params']):
        params_before[idx] = param.clone().detach()

    for epoch in range(1, args.inner_iters_perm + 1):
        for param in optimizer_perm.param_groups[0]['params']:
            param.requires_grad = True
        for param in optimizer_phi.param_groups[0]['params']:
            param.requires_grad = False

        test_res = {'loss': np.nan, 'accuracy': np.nan, 'nll': np.nan, 'loss_time': np.nan, 'acc_time': np.nan}
        time_ep = time.time()

        if args.curve is None or not has_bn or epoch % args.val_freq[0] == 1 or args.val_freq[0] == 1:
            test_res = utils.test_perm(loaders['test'], model, criterion, regularizer=regularizer,
                                       train_loader=loaders['train'], bn_eval=False, samp_t=True)
            idx = scheduler_perm.last_epoch + scheduler_phi.last_epoch
            loss_test[idx] = test_res['loss']
            acc_test[idx] = test_res['accuracy']
            loss_time[idx, :] = test_res['loss_time']
            acc_time[idx, :] = test_res['acc_time']

        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)
        train_res = utils.train_perm(loaders['train'], model, optimizer_perm, scheduler_perm, criterion,
                                     params_old=params_before, regularizer=None, nu=1E3, proj_flag=False,
                                     pen_flag=True, lp_pen=None, tqdm_summary=False)
        scheduler_perm.step()

        time_ep = time.time() - time_ep
        print('Outer Iteration %2d, Permutation Iteration %2d, Training Loss: %.3E, Training Accuracy: %.2f, '
              'Validation Loss: %.3E, Validation Accuracy: %.2f, Time Elapsed: %.2fs' %
              (iter, scheduler_perm.last_epoch, train_res['loss'], train_res['accuracy'], test_res['nll'],
               test_res['accuracy'], time_ep))
        idx = scheduler_perm.last_epoch + scheduler_phi.last_epoch
        loss_train[idx] = train_res['loss']
        acc_train[idx] = train_res['accuracy']
    print('Doubly Stochastic Matrix', optimizer_perm.param_groups[0]['params'][0].data.cpu().detach().numpy())
    utils.sample_permutation(model, optimizer_perm, loaders['train'], loaders['train'], criterion, params_before, k=32)
    print('Permutation Sampled', optimizer_perm.param_groups[0]['params'][0].data.cpu().detach().numpy())
    with torch.no_grad():
        bb = []
        for param, param_o in zip(optimizer_perm.param_groups[0]['params'], params_before):
            bb.append(torch.sum(param * param_o).item())
        print(bb)

    params_before = [None] * len(optimizer_phi.param_groups[0]['params'])
    for idx, param in enumerate(optimizer_phi.param_groups[0]['params']):
        params_before[idx] = param.detach().clone()

    for epoch in range(1, args.inner_iters_phi + 1):
        for param in optimizer_perm.param_groups[0]['params']:
            param.requires_grad = False
        for param in optimizer_phi.param_groups[0]['params']:
            param.requires_grad = True

        test_res = {'loss': np.nan, 'accuracy': np.nan, 'nll': np.nan}
        time_ep = time.time()

        if args.curve is None or not has_bn or epoch % args.val_freq[1] == 1 or args.val_freq[1] == 1:
            test_res = utils.test_perm(loaders['test'], model, criterion, regularizer=regularizer,
                                       train_loader=loaders['train'], bn_eval=False, samp_t=True)
            idx = scheduler_perm.last_epoch + scheduler_phi.last_epoch
            loss_test[idx] = test_res['loss']
            acc_test[idx] = test_res['accuracy']
            loss_time[idx, :] = test_res['loss_time']
            acc_time[idx, :] = test_res['acc_time']

        train_res = utils.train_perm(loaders['train'], model, optimizer_phi, scheduler_phi, criterion,
                                     params_old=params_before, regularizer=regularizer, nu=1E3)
        scheduler_phi.step()

        time_ep = time.time() - time_ep

        print('Outer Iteration %2d, Curve Iteration %2d, Training Loss: %.3E, Training Accuracy: %.2f, '
              'Validation Loss: %.3E, Validation Accuracy: %.2f, Time Elapsed: %.2fs' %
              (iter, scheduler_phi.last_epoch, train_res['loss'], train_res['accuracy'], test_res['nll'],
               test_res['accuracy'], time_ep))
        idx = scheduler_perm.last_epoch + scheduler_phi.last_epoch
        loss_train[idx] = train_res['loss']
        acc_train[idx] = train_res['accuracy']

test_res = utils.test_perm(loaders['test'], model, criterion, regularizer, train_loader=loaders['train'], bn_eval=False,
                           samp_t=True)
loss_test[idx] = test_res['loss']
acc_test[idx] = test_res['accuracy']
loss_time[idx, :] = test_res['loss_time']
acc_time[idx, :] = test_res['acc_time']

if args.model == 'GoogLeNet':
    pam_perm = []
    for perm in model.permutations[1:-1]:
        if len(perm) == 1:
            pam_perm.append(perm[0].cpu().numpy()) 
        else: 
            sub_list = [] 
            for sub_perm in perm:
                sub_list.append(sub_perm.cpu().numpy())
            pam_perm.append(sub_list)
else:
    pam_perm = [torch.nonzero(i)[:, 1].cpu().numpy() for i in model.permutations]
    pam_perm = pam_perm[1:-1]
    if matching is not None:
        pam_perm = [match_og[match_perm] for (match_og, match_perm) in zip(matching, pam_perm)]

model.export_base_parameters(model_turningpt, 1)
model_turningpt_fin = copy.deepcopy(model_turningpt)
if args.model == 'GoogLeNet':
    model.weight_permutation(model_1)
else:
    model.weight_permutation()
model.init_linear()
model.export_base_parameters(model_turningpt, 1)
for param_0, param_1 in zip(model_turningpt_fin.parameters(), model_turningpt.parameters()):
    param_0.data += param_1.data

utils.save_checkpoint(
    args.dir,
    total_iters,
    name='checkpoint_align_pam_%s_seeds_%02d_%02d' % (args.alignment, args.seed_a, args.seed_b),
    model_state=model_turningpt_fin.state_dict(),
    optimizer_state_perm=optimizer_perm.state_dict(),
    optimizer_state_phi=optimizer_phi.state_dict()
)

np.save('%smatch_pam_%s_seeds_%02d_%02d.npy' % (args.dir_alignment, args.alignment, args.seed_a, args.seed_b), pam_perm)

curve_data = {'acc_train': acc_train, 'loss_train': loss_train, 'acc_test': acc_test, 'loss_test': loss_test,
              'iters_perm': args.inner_iters_perm, 'iters_phi': args.inner_iters_phi,
              'loss_time': loss_time, 'acc_time': acc_time, 'change_perm': change_P}
np.save('%scurve_align_pam_%s_seeds_%02d_%02d.npy' % (args.dir2, args.alignment, args.seed_a, args.seed_b), curve_data)
