import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import definitions


parser = argparse.ArgumentParser(description='Displays the plane of intializations for training curves.')
parser.add_argument('--dir', type=str, default='model_data/loss_planes_with_alignment/', metavar='DIR',
                    help='directory for loss plane data (default: model_data/paired_models/)')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--model', nargs='+', type=str, default=['TinyTen', 'ResNet32', 'GoogLeNet'], metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--seed_a', nargs='+', type=int, default=[1, 3, 5], metavar='SEEDa',
                    help='seed init of model 0 (default: 0)')
parser.add_argument('--seed_b', nargs='+', type=int, default=[2, 4, 6], metavar='SEEDb',
                    help='seed init of model 1 (default: 1)')
parser.add_argument('--alignment_type', nargs='+', type=str, default=[''],
                    help='the type of alignment that was used to obtain the third optima (default: None)')
parser.add_argument('--interp', type=str, default='bicubic', metavar='INTERP',
                    help='Interpolation method used for plotting loss plane (default: bicubic)')
args = parser.parse_args()


project_root = definitions.get_project_root()
os.chdir(project_root)

num_alignments = len(args.alignment_type)
num_planes = len(args.seed_a)
if num_planes == 1:
    ddof = 0
else:
    ddof = 1

fig1 = plt.figure(1, figsize=[1.75 * 6.4, 0.8 * 4.8])
fig2 = plt.figure(2, figsize=[1.75 * 6.4, 0.8 * 4.8])
plt.suptitle('Standard Error of Loss \n %s' % args.dataset)
fig3 = plt.figure(3, figsize=[1.75 * 6.4, 0.8 * 4.8])
fig4 = plt.figure(4, figsize=[1.75 * 6.4, 0.8 * 4.8])
plt.suptitle('Standard Error of Accuracy \n %s' % args.dataset)

cmap = plt.cm.jet
for model_idx, model in enumerate(args.model):
    dir_model = ('%s%s/%s/' % (args.dir, model, args.dataset))

    t1 = np.zeros([num_alignments, num_planes, 2])
    t2 = np.zeros([num_alignments, num_planes, 2])
    model1_coords = np.zeros([num_alignments, num_planes, 2])
    model2_coords = np.zeros([num_alignments, num_planes, 2])
    loss = [None] * num_alignments
    acc = [None] * num_alignments
    for j, alignment in enumerate(args.alignment_type):
        for i, (seed_a, seed_b) in enumerate(zip(args.seed_a, args.seed_b)):
            loss_plane = np.load('%s%s_loss_plane_dict_seeds_%02d_%02d.npy' %
                                 (dir_model, alignment, seed_a, seed_b), allow_pickle=True)
            loss_plane = loss_plane[()]
            t1_item = loss_plane['t1']
            t2_item = loss_plane['t2']
            t1[j, i, :] = [np.min(t1_item), np.max(t1_item)]
            t2[j, i, :] = [np.min(t2_item), np.max(t2_item)]
            model1_coords[j, i, :] = loss_plane['model1_coords']
            model2_coords[j, i, :] = loss_plane['model2_coords']
            if loss[j] is None:
                loss[j] = np.zeros([num_planes, len(t1_item), len(t2_item)])
                acc[j] = np.zeros([num_planes, len(t1_item), len(t2_item)])
            loss[j][i] = loss_plane['loss']
            acc[j][i] = loss_plane['acc']

    t1 = np.mean(t1, axis=1)
    t2 = np.mean(t2, axis=1)
    model1_coords = np.mean(model1_coords, axis=1)
    model2_coords = np.mean(model2_coords, axis=1)
    loss = np.asarray(loss)
    acc = np.asarray(acc)

    loss_mu = np.mean(loss, axis=1)
    loss_sigma = np.std(loss, axis=1, ddof=ddof)
    loss_stderr = loss_sigma / num_planes ** 0.5
    acc_mu = np.mean(acc, axis=1)
    acc_sigma = np.std(acc, axis=1, ddof=ddof)
    acc_stderr = acc_sigma / num_planes ** 0.5

    loss_mu_extent = [np.min(loss_mu), np.max(loss_mu)]
    loss_std_extent = [np.min(loss_sigma), np.max(loss_sigma)]
    acc_mu_extent = [np.min(acc_mu), np.max(acc_mu)]
    acc_std_extent = [np.min(acc_sigma), np.max(acc_sigma)]

    alignment_dict = {'corr': 'Correlation', 'rand': 'Random'}

    plt.figure(1)
    plt.subplot(1, len(args.model), model_idx + 1)
    for j, alignment in enumerate(args.alignment_type):
        plt.imshow(loss_mu[j].T, cmap=cmap, interpolation=args.interp, vmin=loss_mu_extent[0],
                   vmax=loss_mu_extent[1], origin='lower', extent=[t1[j, 0], t1[j, 1], t2[j, 0], t2[j, 1]])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('%s' % model, fontsize='xx-large')
        plt.xlabel(r'$\theta_2 - \theta_1$', fontsize='xx-large')
        plt.ylabel(r'$(\theta_2 - \theta_1)^{\perp}$', fontsize='xx-large')
        plt.annotate(s=r'$\theta_1$', xy=(0, 0), xytext=(3, 0.5), color='white', fontsize='x-large')
        plt.annotate(s=r'$\theta_2$', xy=(model1_coords[j, 0], model1_coords[j, 1]),
                     xytext=(-3 + model1_coords[j, 0], 0.5 + model1_coords[j, 1]), color='white',
                     horizontalalignment='right', fontsize='x-large')
        plt.annotate(s=r'$P \theta_2$', xy=(model2_coords[j, 0], model2_coords[j, 1]),
                     xytext=(3 + model2_coords[j, 0], model2_coords[j, 1]), color='white', verticalalignment='top', fontsize='x-large')
        plt.scatter(0, 0, c='white', marker='o')
        plt.scatter(model1_coords[j, 0], model1_coords[j, 1], c='white', marker='o')
        plt.scatter(model2_coords[j, 0], model2_coords[j, 1], c='white', marker='o')
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')
        plt.axis('tight')

    plt.figure(2)
    plt.subplot(1, len(args.model), model_idx + 1)
    for j, alignment in enumerate(args.alignment_type):
        plt.imshow(loss_sigma[j].T, cmap=cmap, interpolation=args.interp, vmin=loss_std_extent[0],
                   vmax=loss_std_extent[1], origin='lower', extent=[t1[j, 0], t1[j, 1], t2[j, 0], t2[j, 1]])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('%s' % model, fontsize='x-large')
        plt.xlabel(r'$\theta_1 - \theta_0$')
        plt.ylabel(r'$(\theta_1 - \theta_0)^{\perp}$')
        plt.annotate(s='Model 0', xy=(0, 0), xytext=(3, 0.5), color='white')
        plt.annotate(s='Model 1', xy=(model1_coords[j, 0], model1_coords[j, 1]),
                     xytext=(-3 + model1_coords[j, 0], 0.5 + model1_coords[j, 1]), color='white',
                     horizontalalignment='right')
        plt.annotate(s='Model 1 Aligned', xy=(model2_coords[j, 0], model2_coords[j, 1]),
                     xytext=(3 + model2_coords[j, 0], model2_coords[j, 1]), color='white', verticalalignment='top')
        plt.scatter(0, 0, c='white', marker='o')
        plt.scatter(model1_coords[j, 0], model1_coords[j, 1], c='white', marker='o')
        plt.scatter(model2_coords[j, 0], model2_coords[j, 1], c='white', marker='o')

    plt.figure(3)
    plt.subplot(1, len(args.model), model_idx + 1)
    for j, alignment in enumerate(args.alignment_type):
        plt.imshow(acc_mu[j].T, cmap=cmap, interpolation=args.interp, vmin=acc_mu_extent[0],
                   vmax=acc_mu_extent[1], origin='lower', extent=[t1[j, 0], t1[j, 1], t2[j, 0], t2[j, 1]])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('%s' % model, fontsize='xx-large')
        plt.xlabel(r'$\theta_2 - \theta_1$', fontsize='xx-large')
        plt.ylabel(r'$(\theta_2 - \theta_1)^{\perp}$', fontsize='xx-large')
        plt.annotate(s=r'$\theta_1$', xy=(0, 0), xytext=(3, 0.5), color='white', fontsize='x-large')
        plt.annotate(s=r'$\theta_2$', xy=(model1_coords[j, 0], model1_coords[j, 1]),
                     xytext=(-3 + model1_coords[j, 0], 0.5 + model1_coords[j, 1]), color='white',
                     horizontalalignment='right', fontsize='x-large')
        plt.annotate(s=r'$P \theta_2$', xy=(model2_coords[j, 0], model2_coords[j, 1]),
                     xytext=(3 + model2_coords[j, 0], 0.5 + model2_coords[j, 1]), color='white',
                     verticalalignment='top', fontsize='x-large')
        plt.scatter(0, 0, c='black', marker='o')
        plt.scatter(model1_coords[j, 0], model1_coords[j, 1], c='black', marker='o')
        plt.scatter(model2_coords[j, 0], model2_coords[j, 1], c='black', marker='o')
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')
        plt.axis('tight')

    plt.figure(4)
    plt.subplot(1, len(args.model), model_idx + 1)
    for j, alignment in enumerate(args.alignment_type):
        plt.imshow(acc_sigma[j].T, cmap=cmap, interpolation=args.interp, vmin=acc_std_extent[0],
                   vmax=acc_std_extent[1], origin='lower', extent=[t1[j, 0], t1[j, 1], t2[j, 0], t2[j, 1]])
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('%s' % model, fontsize='x-large')
        plt.xlabel(r'$\theta_1 - \theta_0$')
        plt.ylabel(r'$(\theta_1 - \theta_0)^{\perp}$')
        plt.annotate(s='Model 0', xy=(0, 0), xytext=(3, 0.5), color='white')
        plt.annotate(s='Model 1', xy=(model1_coords[j, 0], model1_coords[j, 1]),
                     xytext=(-3 + model1_coords[j, 0], 0.5 + model1_coords[j, 1]), color='white',
                     horizontalalignment='right')
        plt.annotate(s='Model 1 Aligned', xy=(model2_coords[j, 0], model2_coords[j, 1]),
                     xytext=(3 + model2_coords[j, 0], 0.5 + model2_coords[j, 1]), color='white', verticalalignment='top')
        plt.scatter(0, 0, c='white', marker='o')
        plt.scatter(model1_coords[j, 0], model1_coords[j, 1], c='white', marker='o')
        plt.scatter(model2_coords[j, 0], model2_coords[j, 1], c='white', marker='o')

plt.figure(1)
plt.tight_layout()
plt.savefig('figures/fin_plane_loss_%s.png' % args.dataset, bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.figure(2)
plt.tight_layout()
plt.figure(3)
plt.tight_layout()
plt.savefig('figures/fin_plane_acc_%s.png' % args.dataset, bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.figure(4)
plt.tight_layout()

# plt.savefig('figures/fin_fft.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.show()
