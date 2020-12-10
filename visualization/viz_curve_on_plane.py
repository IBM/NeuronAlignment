import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import definitions


parser = argparse.ArgumentParser(description='Displays the learned curve on the plane.')
parser.add_argument('--dir_planes', type=str, default='model_data/loss_planes_with_curve/', metavar='DIRPLANES',
                    help='directory for saved evaluated planes (default: model_data/loss_planes_with_curve/)')
parser.add_argument('--dir_curves', type=str, default='model_data/curves/', metavar='DIRCURVES',
                    help='directory for saved evaluated curves (default: model_data/curves/)')
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
parser.add_argument('--epochs', type=int, nargs='+', default=[250, 250, 250], metavar='S',
                    help='Number of epochs the curve was trained for')
parser.add_argument('--interp', type=str, default='bicubic', metavar='INTERP',
                    help='Interpolation method used for plotting loss plane (default: bicubic)')
args = parser.parse_args()


project_root = definitions.get_project_root()
os.chdir(project_root)

num_models = len(args.model)
num_alignments = len(args.alignment)
num_planes = len(args.seed_a)
if num_planes == 1:
    ddof = 0
else:
    ddof = 1

if args.alignment is None:
    args.alignment = ['null']
alignment_dict = {'corr': 'Aligned', 'null': 'Unaligned'}

fig1 = plt.figure(1, figsize=[1.75 * 6.4, 1.75 * 4.8])
# plt.suptitle('Mean Validation Log Loss \n %s' % args.dataset)
fig2 = plt.figure(2, figsize=[1.75 * 6.4, 1.75 * 4.8])
# plt.suptitle('Standard Deviation of Log Loss \n %s' % args.dataset)
fig3 = plt.figure(3, figsize=[1.75 * 6.4, 1.75 * 4.8])
# plt.suptitle('Mean Validation Accuracy \n %s' % args.dataset)
fig4 = plt.figure(4, figsize=[1.75 * 6.4, 1.75 * 4.8])
# plt.suptitle('Standard Deviation of Accuracy \n %s' % args.dataset)
for model_idx, model in enumerate(args.model):
    epoch = args.epochs[model_idx]
    dir_planes = ('%s%s/%s/' % (args.dir_planes, model, args.dataset))
    dir_curves = ('%s%s/%s/' % (args.dir_curves, model, args.dataset))

    curve = [None] * num_alignments
    t1 = np.zeros([num_alignments, num_planes, 2])
    t2 = np.zeros([num_alignments, num_planes, 2])
    model1_coords = np.zeros([num_alignments, num_planes, 2])
    loss = [None] * num_alignments
    acc = [None] * num_alignments
    for j, alignment in enumerate(args.alignment):
        if alignment == '':
            alignment = 'null'
        for i, (seed_a, seed_b) in enumerate(zip(args.seed_a, args.seed_b)):
            curve_eval = np.load('%scurve_align_%s_seeds_%02d_%02d-%d.npy' %
                                 (dir_curves, alignment, seed_a, seed_b, epoch), allow_pickle=True)
            curve_eval = curve_eval[()]

            loss_plane = np.load('%s%s_loss_plane_dict_seeds_%02d_%02d-%d.npy' %
                                 (dir_planes, alignment, seed_a, seed_b, epoch), allow_pickle=True)
            loss_plane = loss_plane[()]
            t1_item = loss_plane['t1']
            t2_item = loss_plane['t2']
            t1[j, i, :] = [np.min(t1_item), np.max(t1_item)]
            t2[j, i, :] = [np.min(t2_item), np.max(t2_item)]
            model1_coords[j, i, :] = loss_plane['model1_coords']
            if loss[j] is None:
                curve[j] = np.zeros([num_planes, len(curve_eval['curve_len']), 2])
                loss[j] = np.zeros([num_planes, len(t1_item), len(t2_item)])
                acc[j] = np.zeros([num_planes, len(t1_item), len(t2_item)])
            curve[j][i] = curve_eval['curve_coords']
            loss[j][i] = loss_plane['loss']
            acc[j][i] = loss_plane['acc']

    curve = np.stack(curve)
    curve = np.mean(curve, axis=1)

    t1 = np.mean(t1, axis=1)
    t2 = np.mean(t2, axis=1)
    model1_coords = np.mean(model1_coords, axis=1)
    loss = np.asarray(loss)
    acc = np.asarray(acc)

    loss_mu = np.mean(loss, axis=1)
    loss_sigma = np.std(loss, axis=1, ddof=ddof)
    loss_stderr = loss_sigma / loss.shape[1] ** 0.5
    acc_mu = np.mean(acc, axis=1)
    acc_sigma = np.std(acc, axis=1, ddof=ddof)
    acc_stderr = acc_sigma / acc.shape[1] ** 0.5

    loss_mu_extent = [np.min(loss_mu), np.max(loss_mu)]
    loss_std_extent = [np.min(loss_sigma), np.max(loss_sigma)]
    acc_mu_extent = [np.min(acc_mu), np.max(acc_mu)]
    acc_std_extent = [np.min(acc_sigma), np.max(acc_sigma)]

    cmap = plt.cm.jet
    plt.figure(1)
    for j, alignment in enumerate(args.alignment):
        plt.subplot(num_alignments, num_models, num_models * j + model_idx + 1)
        plt.imshow(loss_mu[j].T, cmap=cmap, interpolation=args.interp, vmin=loss_mu_extent[0],
                   vmax=loss_mu_extent[1], origin='lower', extent=[t1[j, 0], t1[j, 1], t2[j, 0], t2[j, 1]])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('%s \n %s Endpoints' % (model, alignment_dict[alignment]), fontsize='x-large')
        plt.xlabel(r'$\theta_2 - \theta_1$', fontsize='x-large')
        plt.ylabel(r'$(\theta_2 - \theta_1)^{\perp}$', fontsize='x-large')
        plt.annotate(s='Model 1', xy=(0, 0), xytext=(3, 0.5), color='white', fontsize='large')
        plt.annotate(s='Model 2', xy=(model1_coords[j, 0], model1_coords[j, 1]),
                     xytext=(-3 + model1_coords[j, 0], 0.5 + model1_coords[j, 1]), color='white',
                     horizontalalignment='right', fontsize='large')
        plt.scatter(0, 0, c='white', marker='o')
        plt.scatter(model1_coords[j, 0], model1_coords[j, 1], c='white', marker='o')
        plt.plot(curve[j, :, 0], curve[j, :, 1], 'w-')
        plt.axis('tight')

    plt.figure(2)
    for j, alignment in enumerate(args.alignment):
        plt.subplot(num_alignments, num_models, num_models * j + model_idx + 1)
        plt.imshow(loss_sigma[j].T, cmap=cmap, interpolation=args.interp, vmin=loss_std_extent[0],
                   vmax=loss_std_extent[1], origin='lower', extent=[t1[j, 0], t1[j, 1], t2[j, 0], t2[j, 1]])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('%s \n %s Endpoints' % (model, alignment_dict[alignment]))
        plt.xlabel(r'$\theta_1 - \theta_0$')
        plt.ylabel(r'$(\theta_1 - \theta_0)^{\perp}$')
        plt.annotate(s='Model 0', xy=(0, 0), xytext=(3, 0.5), color='white')
        plt.annotate(s='Model 1', xy=(model1_coords[j, 0], model1_coords[j, 1]),
                     xytext=(-3 + model1_coords[j, 0], 0.5 + model1_coords[j, 1]), color='white',
                     horizontalalignment='right')
        plt.scatter(0, 0, c='white', marker='o')
        plt.scatter(model1_coords[j, 0], model1_coords[j, 1], c='white', marker='o')
        plt.plot(curve[j, :, 0], curve[j, :, 1], 'w-')

    plt.figure(3)
    for j, alignment in enumerate(args.alignment):
        plt.subplot(num_alignments, num_models, num_models * j + model_idx + 1)
        plt.imshow(acc_mu[j].T, cmap=cmap, interpolation=args.interp, vmin=acc_mu_extent[0],
                   vmax=acc_mu_extent[1], origin='lower', extent=[t1[j, 0], t1[j, 1], t2[j, 0], t2[j, 1]])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('%s \n %s Endpoints' % (model, alignment_dict[alignment]), fontsize='x-large')
        plt.xlabel(r'$\theta_2 - \theta_1$', fontsize='x-large')
        plt.ylabel(r'$(\theta_2 - \theta_1)^{\perp}$', fontsize='x-large')
        plt.annotate(s='Model 1', xy=(0, 0), xytext=(3, 0.5), color='white', fontsize='large')
        plt.annotate(s='Model 2', xy=(model1_coords[j, 0], model1_coords[j, 1]),
                     xytext=(-3 + model1_coords[j, 0], 0.5 + model1_coords[j, 1]), color='white',
                     horizontalalignment='right', fontsize='large')
        plt.scatter(0, 0, c='black', marker='o')
        plt.scatter(model1_coords[j, 0], model1_coords[j, 1], c='black', marker='o')
        plt.plot(curve[j, :, 0], curve[j, :, 1], 'w-')

        plt.axis('tight')

    plt.figure(4)
    for j, alignment in enumerate(args.alignment):
        plt.subplot(num_alignments, num_models, num_models * j + model_idx + 1)
        plt.imshow(acc_sigma[j].T, cmap=cmap, interpolation=args.interp, vmin=acc_std_extent[0],
                   vmax=acc_std_extent[1], origin='lower', extent=[t1[j, 0], t1[j, 1], t2[j, 0], t2[j, 1]])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('%s \n %s Endpoints' % (model, alignment_dict[alignment]), fontsize='x-large')
        plt.xlabel(r'$\theta_1 - \theta_0$', fontsize='x-large')
        plt.ylabel(r'$(\theta_1 - \theta_0)^{\perp}$', fontsize='x-large')
        plt.annotate(s='Model 0', xy=(0, 0), xytext=(3, 0.5), color='white', fontsize='large')
        plt.annotate(s='Model 1', xy=(model1_coords[j, 0], model1_coords[j, 1]),
                     xytext=(-3 + model1_coords[j, 0], 0.5 + model1_coords[j, 1]), color='white',
                     horizontalalignment='right', fontsize='large')
        plt.scatter(0, 0, c='white', marker='o')
        plt.scatter(model1_coords[j, 0], model1_coords[j, 1], c='white', marker='o')
        plt.plot(curve[j, :, 0], curve[j, :, 1], 'w-')

plt.figure(1)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('figures/fin_plane_curve_loss_%s.png' % args.dataset, bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.figure(2)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.figure(3)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('figures/fin_plane_curve_acc_%s.png' % args.dataset, bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.figure(4)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
