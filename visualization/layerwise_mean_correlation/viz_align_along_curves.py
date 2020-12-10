import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import definitions


parser = argparse.ArgumentParser(description='Displays correlation signature of curve midpoint to endpoints.')
parser.add_argument('--dir', type=str, default='model_data/paired_models_on_curve/', metavar='DIR',
                    help='directory for saved midpoint matching data (default: /tmp/curve/)')
parser.add_argument('--point_t', type=int, default=0.5, metavar='N',
                    help='the point on the curve to align to each endpoint in the curve parameterization. '
                         '(default: 0.5)')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')

parser.add_argument('--model', nargs='+', type=str, default=['TinyTen', 'ResNet32', 'GoogLeNet'], metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--seed_a', nargs='+', type=int, default=[1, 3, 5], metavar='S',
                    help='random seed for model 0 (default: 1)')
parser.add_argument('--seed_b', nargs='+', type=int, default=[2, 4, 6], metavar='S',
                    help='random seed for model 1(default: 2)')
parser.add_argument('--epochs', type=int, nargs='+', default=[250, 250, 250],
                    help='The number of epochs the curve was trained for. Needed to access checkpoint.')
parser.add_argument('--alignment', type=str, default='corr',
                    help='specify an alignment if the models are to be aligned before curve finding (default: None)')
args = parser.parse_args()


project_root = definitions.get_project_root()
os.chdir(project_root)

# fig = plt.figure(1, figsize=[1.75*6.4, 1.75*4.8])
fig1 = plt.figure(1, figsize=[1.25*6.4, 1.*4.8])
align_dict = {'corr': 'Aligned', 'null': 'Unaligned'}

for model_idx, model in enumerate(args.model):
    epoch = args.epochs[model_idx]
    dir_model = ('%s%s/%s/' % (args.dir, model, args.dataset))

    corr_unaligned_linear_mean = [[], []]
    corr_unaligned_mean = [[], []]
    corr_aligned_mean = [[], []]
    for seed_a, seed_b in zip(args.seed_a, args.seed_b):
        corr_unaligned_linear = np.load('%scorr_unaligned_mean_seeds_%02d_%02d_ends_align_%s-0.npy' %
                                        (dir_model, seed_a, seed_b, args.alignment))
        corr_unaligned = np.load('%scorr_unaligned_mean_seeds_%02d_%02d_ends_align_%s-%d.npy' %
                                 (dir_model, seed_a, seed_b, args.alignment, epoch))
        corr_aligned = np.load('%scorr_aligned_mean_seeds_%02d_%02d_ends_align_%s-%d.npy' %
                               (dir_model, seed_a, seed_b, args.alignment, epoch))
        for i in range(2):
            corr_unaligned_linear_mean[i] += [corr_unaligned_linear[i]]
            corr_unaligned_mean[i] += [corr_unaligned[i]]
            corr_aligned_mean[i] += [corr_aligned[i]]

    corr_unaligned_linear_mean = np.asarray(corr_unaligned_linear_mean)
    corr_unaligned_mean = np.asarray(corr_unaligned_mean)
    corr_aligned_mean = np.asarray(corr_aligned_mean)

    num_models = corr_unaligned_mean.shape[1]
    if num_models > 1:
        ddof = 1
    else:
        ddof = 0

    unaligned_linear_mu = np.mean(corr_unaligned_linear_mean, axis=(0, 1))
    unaligned_linear_sigma = np.std(corr_unaligned_linear_mean, axis=(0, 1), ddof=ddof)
    unaligned_linear_stderr = unaligned_linear_sigma / num_models ** 0.5
    unaligned_mu = np.mean(corr_unaligned_mean, axis=(0, 1))
    unaligned_sigma = np.std(corr_unaligned_mean, axis=(0, 1), ddof=ddof)
    unaligned_stderr = unaligned_sigma / num_models ** 0.5
    aligned_mu = np.mean(corr_aligned_mean, axis=(0, 1))
    aligned_sigma = np.std(corr_aligned_mean, axis=(0, 1), ddof=ddof)
    aligned_stderr = aligned_sigma / num_models ** 0.5

    x = [i+1 for i in range(unaligned_mu.shape[0])]


    plt.subplot(1, len(args.model), model_idx + 1)
    if num_models > 1:
        plt.errorbar(x, unaligned_mu, yerr=unaligned_sigma,
                     label='Before Alignment')
        plt.errorbar(x, aligned_mu, yerr=aligned_sigma, label='After Alignment')
        plt.errorbar(x, unaligned_linear_mu, yerr=unaligned_linear_sigma,
                     label='Linear Midpoint')
    else:
        plt.plot(x, unaligned_mu, label='Before Alignment')
        plt.plot(x, aligned_mu, label='After Alignment')
        plt.plot(x, unaligned_linear_mu, label='Linear Midpoint')
    plt.title('%s' % (model), fontsize='xx-large')
    plt.xlabel('Layer', fontsize='xx-large')
    plt.ylabel('Cross-Correlation', fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.legend(fontsize='large')

plt.tight_layout()
# plt.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.show()
