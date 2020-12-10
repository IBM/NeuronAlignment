import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import definitions


parser = argparse.ArgumentParser(description='Aligns two networks using crosscorrelation of activations.')
parser.add_argument('--dir', type=str, default='model_data/paired_models/', metavar='DIR',
                    help='directory for saving paired model data (default: model_data/paired_models/)')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--model', nargs='+', type=str, default=['TinyTen', 'ResNet32', 'GoogLeNet'], metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--seed_a', nargs='+', type=int, default=[1, 3, 5], metavar='S',
                    help='seed init of model 0 (default: 0)')
parser.add_argument('--seed_b', nargs='+', type=int, default=[2, 4, 6], metavar='S',
                    help='seed init of model 1 (default: 1)')
args = parser.parse_args()


project_root = definitions.get_project_root()
os.chdir(project_root)

num_model_types = len(args.model)
num_models = len(args.seed_a)
if num_models > 1:
    ddof = 1
else:
    ddof = 0

# fig1 = plt.figure(1, figsize=[1.5*6.4, 1*4.8])
fig1 = plt.figure(1, figsize=[1.25*6.4, 1*4.8])
for model_idx, model in enumerate(args.model):
    dir_model = ('%s%s/%s/' % (args.dir, model, args.dataset))

    corr_unaligned_mean = []
    corr_aligned_mean = []
    for seed_a, seed_b in zip(args.seed_a, args.seed_b):
        corr_unaligned_mean += [np.load('%scorr_unaligned_mean_seeds_%02d_%02d.npy' % (dir_model, seed_a, seed_b))]
        corr_aligned_mean += [np.load('%scorr_aligned_mean_seeds_%02d_%02d.npy' % (dir_model, seed_a, seed_b))]

    corr_unaligned_mean = np.asarray(corr_unaligned_mean)
    corr_aligned_mean = np.asarray(corr_aligned_mean)

    unaligned_mu = np.mean(corr_unaligned_mean, axis=0)
    unaligned_sigma = np.std(corr_unaligned_mean, axis=0, ddof=ddof)
    unaligned_stderr = unaligned_sigma / num_models ** 0.5
    aligned_mu = np.mean(corr_aligned_mean, axis=0)
    aligned_sigma = np.std(corr_aligned_mean, axis=0, ddof=ddof)
    aligned_stderr = aligned_sigma / num_models ** 0.5

    x = [i+1 for i in range(unaligned_mu.shape[0])]
    plt.subplot(1, num_model_types, model_idx + 1)
    if model == 'TinyTen' or model == 'VGG16' or model == 'ConvFC':
        if num_models > 1:
            plt.errorbar(x, unaligned_mu, yerr=unaligned_sigma, label='Before Alignment')
            plt.errorbar(x, aligned_mu, yerr=aligned_sigma, label='After Alignment')
        else:
            plt.plot(x, unaligned_mu, label='Before Alignment')
            plt.plot(x, aligned_mu, label='After Alignment')
        plt.title('%s' % model, fontsize='xx-large')
        plt.xlabel('Layer', fontsize='xx-large')
        plt.ylabel('Cross-Correlation', fontsize='xx-large')
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')
        plt.legend(fontsize='x-large')

    elif model == 'ResNet32':
        num_layer = 32
        if num_models > 1:
            plt.errorbar(x, unaligned_mu, unaligned_sigma, label='Before Alignment')
            plt.errorbar(x, aligned_mu, aligned_sigma, label='After Alignment')
        else:
            plt.plot(x, unaligned_mu, label='Before Alignment')
            plt.plot(x, aligned_mu, label='After Alignment')
        plt.title('%s' % model, fontsize='xx-large')
        plt.xlabel('Layer', fontsize='xx-large')
        plt.ylabel('Cross-Correlation', fontsize='xx-large')
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')
        plt.legend(fontsize='x-large')
        # plt.tight_layout()
    else:
        raise ValueError('Not a valid neural network model name. Needs to be either TinyTen, VGG16, or ResNet32.')

plt.figure(1)
plt.tight_layout()
# plt.tight_layout(rect=[0, 0.05, 1, 0.90])
# plt.savefig('figures/fin_corr.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()
