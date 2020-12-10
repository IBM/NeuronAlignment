import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nfft import nfft_adjoint, ndft
from matplotlib.ticker import StrMethodFormatter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import definitions


parser = argparse.ArgumentParser(description='Displays the test accuracy of the adversarial curve between two models.')

parser.add_argument('--dir', type=str, default='model_data/adversarial_curves/', metavar='DIR',
                    help='directory for saved evaluated curves (default: /model_data/adversarial_curves/)')
parser.add_argument('--dir_training', type=str, default='model_data/training/adversarial_curve_models/', metavar='DIR',
                    help='directory for saved curve training data '
                         '(default: model_data/training/adversarial_curve_models/)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--model', nargs='+', type=str, default=['TinyTen', 'ResNet32', 'GoogLeNet'], metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--seed_a', nargs='+', type=int, default=[1, 3, 5], metavar='S',
                    help='random seed for model 0 (default: 1)')
parser.add_argument('--seed_b', nargs='+',
                    type=int, default=[2, 4, 6], metavar='S', help='random seed for model 1(default: 2)')
parser.add_argument('--alignment', nargs='+', type=str, default=['corr', 'null'],
                    help='specify an alignment if the models are to be aligned before curve finding (default: None)')
parser.add_argument('--epochs',
                    type=int, nargs='+', default=[200, 200, 20], metavar='S',
                    help='Number of epochs the curve was trained for')
args = parser.parse_args()


project_root = definitions.get_project_root()
os.chdir(project_root)

num_models = len(args.model)
num_alignments = len(args.alignment)
num_curves = len(args.seed_a)
if num_curves == 1:
    ddof = 0
else:
    ddof = 1

if args.alignment is None:
    args.alignment = ['null']

align_dict = {'null': 'Unaligned', 'corr': 'Aligned', 'pam_': 'PAM Unaligned', 'corr_quad': 'Quadratic Assignment',
              'pam_corr': 'PAM Aligned'}
align_dict2 = {'null': 'Val: Unaligned', 'corr': 'Val: Aligned', 'pam_': 'Val: PAM Unaligned',
               'corr_quad': 'Val: Quadratic Assignment', 'pam_corr': 'Val: PAM Aligned'}
epoch_dict = {'0': 'Linear Interpolation', '250': 'Trained Bezier Curve', '200': 'Trained Bezier Curve', '240': 'PAM'}

fig1 = plt.figure(1, figsize=[1.75 * 6.4, 1.0 * 4.8])
fig2 = plt.figure(2, figsize=[1.75 * 6.4, 1.0 * 4.8])
fig3 = plt.figure(3, figsize=[1.5 * 6.4, 1.2 * 4.8])
fig4 = plt.figure(4, figsize=[1.5 * 6.4, 1.2 * 4.8])
fig5 = plt.figure(5, figsize=[1.5 * 6.4, 1 * 4.8])
fig6 = plt.figure(6, figsize=[1.75 * 6.4, 1.0 * 4.8])
fig7 = plt.figure(7, figsize=[1.75 * 6.4, 1.0 * 4.8])

for model_idx, model in enumerate(args.model):
    epoch = args.epochs[model_idx]
    dir_model = ('%s%s/%s/' % (args.dir, model, args.dataset))
    dir_training = ('%s%s/%s/' % (args.dir_training, model, args.dataset))

    curve_len = [None] * num_alignments
    loss = [None] * num_alignments
    acc = [None] * num_alignments
    loss_robust = [None] * num_alignments
    acc_robust = [None] * num_alignments
    loss_t_train = [None] * num_alignments
    acc_t_train = [None] * num_alignments
    loss_t_test = [None] * num_alignments
    acc_t_test = [None] * num_alignments

    loss_time = [None] * num_alignments
    acc_time = [None] * num_alignments

    loss_line_integral = np.zeros([num_alignments, num_curves])
    acc_line_integral = np.zeros([num_alignments, num_curves])
    loss_r_line_integral = np.zeros([num_alignments, num_curves])
    acc_min = np.zeros([num_alignments, num_curves])
    acc_max = np.zeros([num_alignments, num_curves])

    acc_min_r = np.zeros([num_alignments, num_curves])
    acc_max_r = np.zeros([num_alignments, num_curves])
    for j, alignment in enumerate(args.alignment):
        if alignment == '':
            alignment = 'null'
        for i, (seed_a, seed_b) in enumerate(zip(args.seed_a, args.seed_b)):
            curve_dict = np.load('%scurve_align_%s_seeds_%02d_%02d-%d.npy' %
                                 (dir_model, alignment, seed_a, seed_b, epoch), allow_pickle=True)
            curve_dict = curve_dict[()]
            if loss[j] is None:
                curve_len[j] = np.zeros([num_curves, 33])
                loss[j] = np.zeros([num_curves, 33])
                acc[j] = np.zeros([num_curves, 33])
                loss_robust[j] = np.zeros([num_curves, 33])
                acc_robust[j] = np.zeros([num_curves, 33])
            if len(curve_dict['curve_len']) == 65:
                curve_len[j][i] = curve_dict['curve_len'][::2]
                loss[j][i] = curve_dict['test_loss'][::2]
                acc[j][i] = curve_dict['test_acc'][::2]
                loss_robust[j][i] = curve_dict['test_loss_robust'][::2]
                acc_robust[j][i] = curve_dict['test_acc_robust'][::2]
            else:
                curve_len[j][i] = curve_dict['curve_len']
                loss[j][i] = curve_dict['test_loss']
                acc[j][i] = curve_dict['test_acc']
                loss_robust[j][i] = curve_dict['test_loss_robust']
                acc_robust[j][i] = curve_dict['test_acc_robust']

            curve_train_dict = np.load('%scurve_align_%s_seeds_%02d_%02d.npy' %
                                       (dir_training, alignment, seed_a, seed_b), allow_pickle=True)
            curve_train_dict = curve_train_dict[()]

            if loss_t_train[j] is None:
                loss_t_train[j] = np.zeros([num_curves, len(curve_train_dict['loss_train'])])
                acc_t_train[j] = np.zeros([num_curves, len(curve_train_dict['loss_train'])])
                loss_t_test[j] = np.zeros([num_curves, len(curve_train_dict['loss_train'])])
                acc_t_test[j] = np.zeros([num_curves, len(curve_train_dict['loss_train'])])

                loss_time[j] = [None] * num_curves
                acc_time[j] = [None] * num_curves

            loss_t_train[j][i] = curve_train_dict['loss_train']
            acc_t_train[j][i] = curve_train_dict['acc_train']
            loss_t_test[j][i] = curve_train_dict['loss_test']
            acc_t_test[j][i] = curve_train_dict['acc_test']

            loss_line_integral[j, i] = curve_dict['loss_line_integral']
            x = np.pad(curve_len[j][i], (1, 1), 'edge')
            acc_line_integral[j, i] = np.sum(((x[1:-1] - x[:-2])/2 + (x[2:] - x[1:-1])/2) * acc[j][i]) / x[-1]
            loss_r_line_integral[j, i] = np.sum(((x[1:-1] - x[:-2]) / 2 + (x[2:] - x[1:-1]) / 2) * loss_robust[j][i]) \
                                         / x[-1]

            acc_min_r[j, i] = np.min(acc_robust[j][i])
            acc_max_r[j, i] = np.max(acc_robust[j][i])

            acc_min[j, i] = acc[j][i][np.argmin(acc_robust[j][i])]
            acc_max[j, i] = acc[j][i][np.argmax(acc_robust[j][i])]

    loss_time = np.asarray(loss_time)

    loss_fft = np.fft.rfft(loss_time, axis=-1).real

    curve_len = np.stack(curve_len)
    loss = np.stack(loss)
    loss_robust = np.stack(loss_robust)
    loss_fft = np.zeros([loss.shape[0], loss.shape[1], loss.shape[2] - 1])
    for i in range(loss.shape[0]):
        for j in range(loss.shape[1]):
            t_fft = curve_len[i, j] / curve_len[i, j, -1] - 1/2
            loss_fft[i, j] = ndft(t_fft[:-1], loss[i, j, :-1]).real
    loss_fft = loss_fft[:, :, 32:]
    loss_fft = np.real(np.fft.rfft(loss, axis=-1))
    loss_fft = loss_fft[:, :, loss.shape[2] // 2:]
    loss_base = np.zeros_like(loss)
    for i in range(loss.shape[-1]):
        i_t = i / (loss.shape[-1] - 1)
        loss_base[:, :, i] = loss[:, :, i] - ((1 - i_t) * loss[:, :, 0] + i_t * loss[:, :, -1])
    loss_fft = np.fft.rfft(loss, axis=-1)
    loss_fft = loss_fft / np.sum(loss_fft ** 2, axis=-1, keepdims=True) ** 0.5
    loss_fft = np.abs(loss_fft)

    acc = np.stack(acc)
    acc_robust = np.stack(acc_robust)
    loss_t_train = np.stack(loss_t_train)
    acc_t_train = np.stack(acc_t_train)
    loss_t_test = np.stack(loss_t_test)
    acc_t_test = np.stack(acc_t_test)

    curve_len = np.mean(curve_len, axis=1)
    t = np.linspace(0, 1, len(curve_len[0]))

    def compute_stats(signal):
        signal_mu = np.mean(signal, axis=1)
        signal_sigma = np.std(signal, axis=1, ddof=ddof)
        signal_stderr = signal_sigma / signal.shape[1] ** 0.5
        return signal_mu, signal_sigma, signal_stderr

    loss_mu, loss_sigma, loss_stderr = compute_stats(loss)
    acc_mu, acc_sigma, acc_stderr = compute_stats(acc)

    loss_r_mu, loss_r_sigma, loss_r_stderr = compute_stats(loss_robust)
    acc_r_mu, acc_r_sigma, acc_r_stderr = compute_stats(acc_robust)

    loss_fft_mu, loss_fft_sigma, loss_fft_stderr = compute_stats(loss_fft)

    loss_t_train_mu, loss_t_train_sigma, loss_t_train_stderr = compute_stats(loss_t_train)
    acc_t_train_mu, acc_t_train_sigma, acc_t_train_stderr = compute_stats(acc_t_train)
    loss_t_test_mu, loss_t_test_sigma, loss_t_test_stderr = compute_stats(loss_t_test)
    acc_t_test_mu, acc_t_test_sigma, acc_t_test_stderr = compute_stats(acc_t_test)

    loss_line_integral_mu, loss_line_integral_sigma, loss_line_integral_stderr = compute_stats(loss_line_integral)
    loss_r_line_integral_mu, loss_r_line_integral_sigma, loss_r_line_integral_stderr \
        = compute_stats(loss_r_line_integral)
    acc_line_integral_mu, acc_line_integral_sigma, acc_line_integral_stderr = compute_stats(acc_line_integral)

    acc_min_mu, acc_min_sigma, acc_min_stderr = compute_stats(acc_min)
    acc_max_mu, acc_max_sigma, acc_max_stderr = compute_stats(acc_max)

    acc_min_r_mu, acc_min_r_sigma, acc_min_r_stderr = compute_stats(acc_min_r)
    acc_max_r_mu, acc_max_r_sigma, acc_max_r_stderr = compute_stats(acc_max_r)

    print('Model: %s' % model)
    print('Endpoint')
    print('%0.1f pm %0.1f' % (0.5*(acc_mu[0, 0] + acc_mu[0, -1]), 0.5*(acc_sigma[0, 0]**2 + acc_sigma[0, -1]**2)**0.5 ))
    print('%0.2f' % np.maximum(acc_mu[0, 0],acc_mu[0, -1]))
    print('%0.1f pm %0.1f' % (0.5 * (acc_r_mu[0, 0] + acc_r_mu[0, -1]), 0.5 * (acc_r_sigma[0, 0] ** 2 + acc_r_sigma[0, -1] ** 2) ** 0.5))
    print('%0.2f' % np.maximum(acc_r_mu[0, 0], acc_r_mu[0, -1]))
    for j, alignment in enumerate(args.alignment):
        print('Line Integrated Loss (%s): %0.3f +/- %0.3f' %
              (align_dict[alignment], loss_r_line_integral_mu[j], loss_r_line_integral_sigma[j]))
        print('Line Integrated Accuracy (%s): %0.3f +/- %0.3f' %
              (align_dict[alignment], acc_line_integral_mu[j], acc_line_integral_sigma[j]))
        print('Worst accuracy along the curve (%s): %0.3f +/- %0.3f' %
              (align_dict[alignment], acc_min_mu[j], acc_min_sigma[j]))
        print('Best accuracy along the curve (%s): %0.3f +/- %0.3f' %
              (align_dict[alignment], acc_max_mu[j], acc_max_sigma[j]))
        print('Worst robust accuracy along the curve (%s): %0.3f +/- %0.3f' %
              (align_dict[alignment], acc_min_r_mu[j], acc_min_r_sigma[j]))
        print('Best robust accuracy along the curve (%s): %0.3f +/- %0.3f' %
              (align_dict[alignment], acc_max_r_mu[j], acc_max_r_sigma[j]))
        print('Training Loss (%s): %0.3f +/- %0.3f' %
              (align_dict[alignment], loss_t_train_mu[j][-1], loss_t_train_sigma[j][-1]))

    colors = ['blue', 'green', 'red', 'purple']
    markers = ['o', 'v']
    plt.figure(1)
    plt.subplot(1, num_models, model_idx + 1)
    for j, alignment in enumerate(args.alignment):
        if alignment == '':
            alignment = 'null'
        if num_curves > 1:
            plt.errorbar(t, loss_mu[j], loss_sigma[j], label=align_dict[alignment],
                         color=colors[j], marker=markers[j], markevery=8)
        else:
            plt.plot(t, loss_mu[j], label=align_dict[alignment], color=colors[j], marker=markers[j], markevery=8)
    plt.title('%s' % model, fontsize='xx-large')
    plt.xlabel('t', fontsize='xx-large')
    plt.ylabel('Clean Loss', fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))

    plt.legend(fontsize='x-large')

    mask = np.isfinite(loss_t_test_mu[0])
    mask[0] = False
    plt.figure(5)
    plt.subplot(1, num_models, model_idx + 1)
    for j, alignment in enumerate(args.alignment):
        if alignment == '':
            alignment = 'null'
        if num_curves > 1:
            plt.errorbar(np.arange(1, 9), loss_fft_mu[j, 1:9], loss_fft_sigma[j, 1:9],
                         label=align_dict[alignment],
                         color=colors[j])
        else:
            plt.plot(np.arange(1, 9), loss_fft_mu[j, 1:9], label=align_dict[alignment],
                     color=colors[j])
        plt.title('%s' % model)
    plt.xlabel('Wavenumber, k')
    plt.ylabel('|F(L)|')
    plt.legend()

    plt.figure(6)
    plt.subplot(1, num_models, model_idx + 1)
    for j, alignment in enumerate(args.alignment):
        if alignment == '':
            alignment = 'null'
        if num_curves > 1:
            plt.errorbar(t, loss_r_mu[j], loss_r_sigma[j], label=align_dict[alignment],
                         color=colors[j], marker=markers[j], markevery=8)
        else:
            plt.plot(t, loss_r_mu[j], label=align_dict[alignment], color=colors[j], marker=markers[j], markevery=8)
    plt.title('%s' % model, fontsize='xx-large')
    plt.xlabel('t', fontsize='xx-large')
    plt.ylabel('Robust Loss', fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
    plt.legend(fontsize='large')

    plt.figure(2)
    plt.subplot(1, num_models, model_idx + 1)
    for j, alignment in enumerate(args.alignment):
        if alignment == '':
            alignment = 'null'
        if num_curves > 1:
            plt.errorbar(t, acc_mu[j], acc_sigma[j], label=align_dict[alignment],
                         color=colors[j], marker=markers[j], markevery=8)
        else:
            plt.plot(t, acc_mu[j], label=align_dict[alignment], color=colors[j], marker=markers[j], markevery=8)
    plt.title('%s' % model, fontsize='xx-large')
    plt.xlabel('t', fontsize='xx-large')
    plt.ylabel('Clean Accuracy', fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    plt.legend(fontsize='large')

    plt.figure(7)
    plt.subplot(1, num_models, model_idx + 1)
    for j, alignment in enumerate(args.alignment):
        if alignment == '':
            alignment = 'null'
        if num_curves > 1:
            plt.errorbar(t, acc_r_mu[j], acc_r_sigma[j], label=align_dict[alignment],
                         color=colors[j], marker=markers[j], markevery=8)
        else:
            plt.plot(t, acc_r_mu[j], label=align_dict[alignment], color=colors[j], marker=markers[j], markevery=8)
    plt.title('%s' % model, fontsize='xx-large')
    plt.xlabel('t', fontsize='xx-large')
    plt.ylabel('Robust Accuracy', fontsize='xx-large')
    plt.xticks(fontsize='x-large')
    plt.yticks(fontsize='x-large')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    plt.legend(fontsize='large')

    x = np.asarray([i+1 for i in range(len(loss_t_train_mu[0]))])
    mask = np.isfinite(loss_t_test_mu[0])
    plt.figure(3)
    plt.subplot(1, num_models, model_idx + 1)
    for j, alignment in enumerate(args.alignment):
        if alignment == '':
            alignment = 'null'
        if num_curves > 1:
            plt.errorbar(x, loss_t_train_mu[j], loss_t_train_sigma[j],
                         label=align_dict[alignment], color=colors[j], marker=markers[j], markevery=20)
        else:
            plt.plot(x, loss_t_train_mu[j], label=align_dict[alignment], color=colors[j], marker=markers[j], markevery=20)
    plt.title('%s' % model, fontsize='xx-large')
    plt.xlabel('Epoch', fontsize='xx-large')
    plt.ylabel('Loss', fontsize='xx-large')
    plt.legend(fontsize='large')
    x1, x2, y1, y2 = plt.axis()
    y_max = np.min(loss_t_train_mu[:, 1])
    plt.axis((x1, x2, y1, y_max))

    plt.figure(4)
    plt.subplot(1, num_models, model_idx + 1)
    for j, alignment in enumerate(args.alignment):
        if alignment == '':
            alignment = 'null'
        if num_curves > 1:
            plt.errorbar(x, acc_t_train_mu[j], acc_t_train_sigma[j],
                         label=align_dict[alignment], color=colors[j], marker=markers[j], markevery=20)
        else:
            plt.plot(x[mask], acc_t_train_mu[j, mask], label=align_dict[alignment], color=colors[j], marker=markers[j], markevery=20)
    plt.title('%s' % model, fontsize='xx-large')
    plt.xlabel('Epoch', fontsize='xx-large')
    plt.ylabel('Accuracy', fontsize='xx-large')
    plt.legend(fontsize='large')
    x1, x2, y1, y2 = plt.axis()
    y_min = np.max(acc_t_train_mu[:, 5])
    plt.axis((x1, x2, y_min, y2))

plt.figure(1)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('figures/fin_curve_loss.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.figure(2)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('figures/fin_curve_acc.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.figure(3)
plt.tight_layout(rect=[0, 0.03, 1, 0.9])
# plt.savefig('figures/fin_train_loss_stl10.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.figure(4)
plt.tight_layout(rect=[0, 0.03, 1, 0.9])
# plt.savefig('figures/fin_train_acc_stl10.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.figure(5)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('figures/fin_fft.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.figure(6)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('figures/fin_curve_loss_robust.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.figure(7)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('figures/fin_curve_acc_robust.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.show()
