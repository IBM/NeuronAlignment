import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models
from utils import utils, data
import definitions
import random


parser = argparse.ArgumentParser(description='Trains a Typical Neural Network.')

parser.add_argument('--dir', type=str, default='model_dicts/basic_models/', metavar='DIR',
                    help='directory for saving model dicts (default: model_dicts/basic_models/)')
parser.add_argument('--dir2', type=str, default='model_data/training/basic_models/', metavar='DIR',
                    help='directory for saving model training data (default: model_data/training/basic_models/)')
parser.add_argument('--data_path', type=str, default='data/', metavar='PATH',
                    help='path to datasets location (default: data/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: True)')
parser.add_argument('--transform', type=str, default='TinyThree', metavar='TRANSFORM',
                    help='transform name (default: TinyTen)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default='TinyThree', metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--resume', type=str, default=False, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--ckpt', type=str, default='checkpoint-200.pt', metavar='CKPT',
                    help='checkpoint to eval (default: checkpoint-200.pt)')

parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=200, metavar='N',
                    help='save frequency (default: 200)')
parser.add_argument('--val_freq', type=int, default=1, metavar='N',
                    help='save frequency (default: 1)')
parser.add_argument('--lr', type=float, default=1E-1, metavar='LR',
                    help='initial learning rate (default: 1E-1)')
parser.add_argument('--wd', type=float, default=5E-4, metavar='WD',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--lr_decay', type=float, default=0.9996, help='Learning Rate Decay for SGD. (default: 0.9996)')
parser.add_argument('--lr_drop', type=int, default=20, help='Number of epochs required to decay learning rate. '
                                                            '(default: 20)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
args.dir = ('%s%s/%s/' % (args.dir, args.model, args.dataset))
args.dir2 = ('%s%s/%s/' % (args.dir2, args.model, args.dataset))

project_root = definitions.get_project_root()
os.chdir(project_root)
os.makedirs(args.data_path, exist_ok=True)
os.makedirs(args.dir, exist_ok=True)
os.makedirs(args.dir2, exist_ok=True)

print('Arguments')
for arg in vars(args):
    print('%s: %s' % (arg, str(getattr(args, arg))))

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

np.random.seed(args.seed)
random.seed(args.seed)
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
    shuffle_train=True,
    test_batch_size= args.batch_size * 2
)

architecture = getattr(models, args.model)

if args.model == 'VGG16' and args.dataset == 'STL10':
    model = architecture.base(num_classes=num_classes, device=device, dense_pool=3, **architecture.kwargs)
else:
    model = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
model.to(device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
model_params = sum([np.prod(p.size()) for p in model_parameters])
print('Number of Trainable Parameters: %d' % model_params)

if args.resume:
    checkpoint = torch.load(args.ckpt)
    print('Resume training from %d' % checkpoint['epoch'])
    model.load_state_dict(checkpoint['model_state'])

if args.model == 'GoogLeNet':
    criterion = utils.googlenet_criterion
else:
    criterion = nn.CrossEntropyLoss()
regularizer = None

optimizer = optim.SGD(
    filter(lambda param: param.requires_grad, model.parameters()),
    lr=args.lr,
    momentum=0.9,
    weight_decay=args.wd,
    nesterov=True)

# optimizer = optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=1E-3, weight_decay=args.wd)

lambda1 = lambda epoch: 0.5 ** (epoch // args.lr_drop) * args.lr_decay ** epoch
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

start_epoch = 1

test_res = {'loss': None, 'accuracy': None, 'nll': None}
test_loss = np.zeros(args.epochs)
test_acc = np.zeros(args.epochs)
train_loss = np.zeros(args.epochs)
train_acc = np.zeros(args.epochs)
for epoch in range(start_epoch, args.epochs + 1):
    test_res = {'loss': np.nan, 'nll': np.nan, 'accuracy': np.nan}
    time_ep = time.time()
    train_res = utils.train(loaders['train'], model, optimizer, criterion, regularizer, tqdm_summary=False)
    if (epoch - 1) % args.val_freq == 0: 
        test_res = utils.test(loaders['test'], model, criterion, regularizer)

    if epoch % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch,
            name='checkpoint_seed_%02d' % args.seed,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )
    scheduler.step()
    time_ep = time.time() - time_ep

    train_loss[epoch-1] = train_res['loss']
    train_acc[epoch-1] = train_res['accuracy']
    test_loss[epoch-1] = test_res['loss']
    test_acc[epoch-1] = test_res['accuracy']

    print('Epoch %3d, Training Loss: %.3E, Training Accuracy: %.4f, Validation Loss: %.3E, '
          'Validation Accuracy: %.3f, Time Elapsed: %.2fs' %
          (epoch, train_res['loss'], train_res['accuracy'], test_res['nll'], test_res['accuracy'], time_ep))

utils.save_checkpoint(
    args.dir,
    args.epochs,
    name='checkpoint_seed_%02d' % args.seed,
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict()
)

np.save('%sloss_train_seed_%02d' % (args.dir2, args.seed), train_loss)
np.save('%sloss_test_seed_%02d' % (args.dir2, args.seed), test_loss)
np.save('%saccuracy_train_seed_%02d' % (args.dir2, args.seed), train_acc)
np.save('%saccuracy_test_seed_%02d' % (args.dir2, args.seed), test_acc)

print('Done')
