import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import curves
from utils import utils, alignment, data
import models
import definitions


parser = argparse.ArgumentParser(description='Trains a curve between two neural networks.')
parser.add_argument('--dir', type=str, default='model_dicts/curve_models/', metavar='DIR',
                    help='directory for saving curve models (default: model_dicts/curve_models/)')
parser.add_argument('--dir2', type=str, default='model_data/training/curve_models/', metavar='DIR',
                    help='directory for saving curve model data (default: model_data/training/curve_models/)')
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

parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 250)')
parser.add_argument('--save_freq', type=int, default=200, metavar='N',
                    help='save frequency (default: 200)')

parser.add_argument('--lr', type=float, default=1E-1, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--wd', type=float, default=5e-4, metavar='WD',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--lr_decay', type=float, default=0.9996, help='Learning Rate Decay for SGD. (default: 0.9996)')
parser.add_argument('--lr_drop', type=int, default=20, help='Number of epochs required to decay learning rate. '
                                                            '(default: 20)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: None)')
parser.add_argument('--seed_a', type=int, default=None, metavar='S', help='random seed for model 0 (default: None)')
parser.add_argument('--seed_b', type=int, default=None, metavar='S', help='random seed for model 1(default: None)')
parser.add_argument('--epochs_model',
                    type=int, default=200, metavar='EPOCHS', help='Number of epochs the models were trained for')
parser.add_argument('--alignment', type=str, default='',
                    help='specify an alignment if the models are to be aligned before curve finding (default: None)')
parser.add_argument('--val_freq', type=int, default=None,
                    help='the rate in epochs at which to evaluate the model on the validation set.')
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
    test_batch_size=args.batch_size
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

if args.alignment is not None and args.alignment != '':
    matching = np.load('%smatch_%s_seeds_%02d_%02d.npy' %
                       (args.dir_alignment, args.alignment, args.seed_a, args.seed_b), allow_pickle=True)
    if args.model == 'ResNet32':
        model_1, _ = alignment.align_models_resnet(model_1, matching)
    elif args.model == 'GoogLeNet': 
        model_1.align_inception(matching)
    else:
        model_1 = alignment.align_models(model_1, matching)
else:
    args.alignment = 'null'

curve = getattr(curves, args.curve)
model = curves.CurveNet(
    num_classes,
    device,
    curve,
    architecture.curve,
    args.num_bends,
    args.fix_start,
    args.fix_end,
    architecture_kwargs=architecture.kwargs,
)

optimizer = optim.SGD(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=args.lr,
    momentum=0.9,
    weight_decay=args.wd if args.curve is None else 0.0,
    nesterov=True)
lambda1 = lambda epoch: 0.5 ** (epoch // args.lr_drop) * args.lr_decay ** epoch
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

if args.resume is None:
    model.import_base_parameters(model_0, 0)
    model.import_base_parameters(model_1, 2)
    if args.init_linear:
        print('Linear initialization.')
        model.init_linear()
    start_epoch = 1
else:
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print('Resume training from %s' % args.resume)
model.to(device)
model_1.to(device)
model_0.to(device)

model_turningpt = architecture.base(num_classes=num_classes, device=device)
model.export_base_parameters(model_turningpt, 1)

utils.save_checkpoint(
    args.dir,
    0,
    name='checkpoint_align_%s_seeds_%02d_%02d' % (args.alignment, args.seed_a, args.seed_b),
    model_state=model_turningpt.state_dict(),
    optimizer_state=optimizer.state_dict()
)


if args.model == 'GoogLeNet':
    criterion = utils.googlenet_criterion
else:
    criterion = nn.CrossEntropyLoss()
regularizer = None if args.curve is None else curves.l2_regularizer(args.wd)

if args.val_freq is None:
    args.val_freq = np.nan

acc_train = np.ones(args.epochs+1) * np.nan
acc_test = np.ones(args.epochs+1) * np.nan
loss_train = np.ones(args.epochs+1) * np.nan
loss_test = np.ones(args.epochs+1) * np.nan

number_batches = sum(1 for _ in loaders['test'])
loss_time = np.ones([args.epochs+1, number_batches]) * np.nan
acc_time = np.ones([args.epochs+1, number_batches]) * np.nan

has_bn = utils.check_bn(model)
lr = args.lr

for epoch in range(start_epoch, args.epochs + 1):
    test_res = {'loss': np.nan, 'accuracy': np.nan, 'nll': np.nan}
    time_ep = time.time()

    if args.curve is None or not has_bn or epoch % args.val_freq == 1:
        test_res = utils.test(loaders['test'], model, criterion, regularizer,
                              train_loader=loaders['train'], bn_eval=True, samp_t=True)
        loss_test[epoch - 1] = test_res['loss']
        acc_test[epoch - 1] = test_res['accuracy']
        loss_time[epoch - 1, :] = test_res['loss_time']
        acc_time[epoch - 1, :] = test_res['acc_time']

    train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer)
    scheduler.step()

    if epoch % args.save_freq == 0:
        model.export_base_parameters(model_turningpt, 1)
        utils.save_checkpoint(
            args.dir,
            epoch,
            name='checkpoint_align_%s_seeds_%02d_%02d' % (args.alignment, args.seed_a, args.seed_b),
            model_state=model_turningpt.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

    time_ep = time.time() - time_ep

    print('Epoch %3d, Training Loss: %.3E, Training Accuracy: %.4f, Validation Loss: %.3E, '
          'Validation Accuracy: %.3f, Time Elapsed: %.2fs' %
          (epoch, train_res['loss'], train_res['accuracy'], test_res['nll'], test_res['accuracy'], time_ep))
    loss_train[epoch] = train_res['loss']
    acc_train[epoch] = train_res['accuracy']

test_res = utils.test(loaders['test'], model, criterion, regularizer,
                      train_loader=loaders['train'], samp_t=True, bn_eval=True)
loss_test[epoch] = test_res['loss']
acc_test[epoch] = test_res['accuracy']
loss_time[epoch, :] = test_res['loss_time']
acc_time[epoch, :] = test_res['acc_time']

if args.epochs % args.save_freq != 0:
    model.export_base_parameters(model_turningpt, 1)
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        name='checkpoint_align_%s_seeds_%02d_%02d' % (args.alignment, args.seed_a, args.seed_b),
        model_state=model_turningpt.state_dict(),
        optimizer_state=optimizer.state_dict()
    )

curve_data = {'acc_train': acc_train, 'loss_train': loss_train, 'acc_test': acc_test, 'loss_test': loss_test,
              'loss_time': loss_time, 'acc_time': acc_time}
np.save('%scurve_align_%s_seeds_%02d_%02d.npy' % (args.dir2, args.alignment, args.seed_a, args.seed_b), curve_data)
