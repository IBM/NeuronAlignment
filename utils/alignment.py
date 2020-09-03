from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import torch.nn as nn
import copy
from utils import crosscorrelation as cc


class AlignedModelPairs:

    def __init__(self, model0, model1, align_set, adv_flag=False, net0=None, net1=None):
        super(AlignedModelPairs, self).__init__()
        self.model0 = model0
        self.model1 = model1
        self.align_set = align_set
        self.xx_prod = None
        self.yy_prod = None
        self.xy_prod = None
        self.x_mean = None
        self.y_mean = None
        self.cross_cors = None
        self.matches = None
        self.adv_flag = adv_flag
        self.net0 = net0
        self.net1 = net1

    def compute_moments(self):
        self.model0.eval()
        self.model1.eval()
        if self.adv_flag:
            self.net0.eval()
            self.net1.eval()
        xy_cov = None
        xx_cov = None
        yy_cov = None
        x_mean = None
        y_mean = None
        with torch.no_grad():
            for input, target in self.align_set:
                input = input.to(self.model0.device)
                if self.adv_flag:
                    target = target.to(self.model0.device)
                    output0 = self.net0(input, target)[1]
                    output1 = self.net1(input, target)[1]
                    output0 = self.model0(output0, store_int=True)[1]
                    output1 = self.model1(output1, store_int=True)[1]
                else:
                    output0 = self.model0(input, store_int=True)[1]
                    output1 = self.model1(input, store_int=True)[1]
                if xx_cov is None:
                    xy_cov = [None for _ in output0]
                    xx_cov = [None for _ in output0]
                    yy_cov = [None for _ in output0]
                    x_mean = [None for _ in output0]
                    y_mean = [None for _ in output0]
                for idx, (int_act0, int_act1) in enumerate(zip(output0, output1)):
                    if isinstance(int_act0, list):
                        if x_mean[idx] is None:
                            xy_cov[idx] = [np.zeros([lx.shape[0], lx.shape[0]]) for lx in int_act0]
                            xx_cov[idx] = [np.zeros([lx.shape[0], lx.shape[0]]) for lx in int_act0]
                            yy_cov[idx] = [np.zeros([lx.shape[0], lx.shape[0]]) for lx in int_act0]
                            x_mean[idx] = [np.zeros([lx.shape[0], 1]) for lx in int_act0]
                            y_mean[idx] = [np.zeros([lx.shape[0], 1]) for lx in int_act0]
                        for sub_idx, (sub_act0, sub_act1) in enumerate(zip(int_act0, int_act1)):
                            sub_act0 = sub_act0.data.cpu().numpy()
                            sub_act1 = sub_act1.data.cpu().numpy()
                            x_mean[idx][sub_idx] += sub_act0.mean(axis=1, keepdims=True) * input.shape[0] \
                                                    / len(self.align_set.dataset)
                            y_mean[idx][sub_idx] += sub_act1.mean(axis=1, keepdims=True) * input.shape[0] \
                                                    / len(self.align_set.dataset)
                            xy_cov[idx][sub_idx] += np.matmul(sub_act0, sub_act1.transpose()) \
                                      * input.shape[0] / len(self.align_set.dataset)
                            xx_cov[idx][sub_idx] += np.matmul(sub_act0, sub_act0.transpose()) \
                                                    * input.shape[0] / len(self.align_set.dataset)
                            yy_cov[idx][sub_idx] += np.matmul(sub_act1, sub_act1.transpose()) \
                                                    * input.shape[0] / len(self.align_set.dataset)
                    else:
                        if x_mean[idx] is None:
                            xy_cov[idx] = np.zeros([int_act0.shape[0], int_act0.shape[0]])
                            xx_cov[idx] = np.zeros([int_act0.shape[0], int_act0.shape[0]])
                            yy_cov[idx] = np.zeros([int_act0.shape[0], int_act0.shape[0]])
                            x_mean[idx] = np.zeros([int_act0.shape[0], 1])
                            y_mean[idx] = np.zeros([int_act0.shape[0], 1])
                        int_act0 = int_act0.data.cpu().numpy()
                        int_act1 = int_act1.data.cpu().numpy()
                        x_mean[idx] += int_act0.mean(axis=1, keepdims=True) * input.shape[0] \
                                       / len(self.align_set.dataset)
                        y_mean[idx] += int_act1.mean(axis=1, keepdims=True) * input.shape[0] \
                                       / len(self.align_set.dataset)
                        xy_cov[idx] += np.matmul(int_act0, int_act1.transpose()) \
                                       * input.shape[0] / len(self.align_set.dataset)
                        xx_cov[idx] += np.matmul(int_act0, int_act0.transpose()) \
                                       * input.shape[0] / len(self.align_set.dataset)
                        yy_cov[idx] += np.matmul(int_act1, int_act1.transpose()) \
                                       * input.shape[0] / len(self.align_set.dataset)

            self.x_mean = x_mean
            self.y_mean = y_mean
            self.xx_prod = xx_cov
            self.yy_prod = yy_cov
            self.xy_prod = xy_cov

    def compute_crosscorr(self):
        eps = 1E-12
        crosscorr_list = [None for _ in self.x_mean]
        for idx, (x, y, xx, yy, xy) in enumerate(zip(self.x_mean, self.y_mean, self.xx_prod, self.yy_prod,
                                                     self.xy_prod)):
            if isinstance(x, list):
                if crosscorr_list[idx] is None:
                    crosscorr_list[idx] = [np.zeros([x_sub.shape[0], x_sub.shape[0]]) for x_sub in x]
                for idx_sub, (x_sub, y_sub, xx_sub, yy_sub, xy_sub) in enumerate(zip(x, y, xx, yy, xy)):
                    cov_xy = xy_sub - np.matmul(x_sub, y_sub.transpose())
                    cov_xx = xx_sub - np.matmul(x_sub, x_sub.transpose())
                    cov_yy = yy_sub - np.matmul(y_sub, y_sub.transpose())
                    cov_dx = (np.diag(cov_xx) + eps) ** -0.5
                    cov_dx = np.expand_dims(cov_dx, -1)
                    cov_dy = (np.diag(cov_yy) + eps) ** -0.5
                    cov_dy = np.expand_dims(cov_dy, 0)
                    crosscorr_list[idx][idx_sub] = cov_dx * cov_xy * cov_dy
            else:
                cov_xy = xy - np.matmul(x, y.transpose())
                cov_xx = xx - np.matmul(x, x.transpose())
                cov_yy = yy - np.matmul(y, y.transpose())
                cov_dx = (np.diag(cov_xx) + eps) ** -0.5
                cov_dx = np.expand_dims(cov_dx, -1)
                cov_dy = (np.diag(cov_yy) + eps) ** -0.5
                cov_dy = np.expand_dims(cov_dy, 0)
                crosscorr_list[idx] = cov_dx * cov_xy * cov_dy
        self.cross_cors = crosscorr_list

    def compute_match(self):
        matches = [None for _ in self.x_mean]
        for idx, crs_cor in enumerate(self.cross_cors):
            if isinstance(crs_cor, list):
                if matches[idx] is None:
                    matches[idx] = [None for _ in crs_cor]
                for idx_sub, crs_cor_sub in enumerate(crs_cor):
                    hard_match = compute_alignment(crs_cor_sub)
                    matches[idx][idx_sub] = hard_match[:, 1]
                    print('Mean correlation before/after', idx, idx_sub, np.mean(np.diag(crs_cor_sub)),
                          np.mean(np.diag(crs_cor_sub[:, hard_match[:, 1]])))
            else:
                hard_match = compute_alignment(crs_cor)
                matches[idx] = hard_match[:, 1]
                print('Mean correlation before/after', idx, np.mean(np.diag(crs_cor)),
                      np.mean(np.diag(crs_cor[:, hard_match[:, 1]])))
        self.matches = matches


def compute_alignment(corr, neg=True):
    num_filter = corr.shape[0]
    hard_match = np.zeros([num_filter, 2])
    if neg:
        hard_match[:, 0], hard_match[:, 1] = linear_sum_assignment(1.01 - corr)
    else:
        hard_match[:, 0], hard_match[:, 1] = linear_sum_assignment(corr)
    hard_match = hard_match.astype(int)
    return hard_match


def compute_alignment_random(corr, seed=None):
    num_filter = corr.shape[0]
    random_match = np.zeros([num_filter, 2])
    random_match[:, 0] = range(num_filter)
    random_match[:, 1] = np.random.RandomState(seed=seed).permutation(num_filter)
    return random_match


def compute_model_alignment(model0, model1, dataloader, num_layer=None, return_corr=None, quad_assignment=False,
                            use_warmstart=True):
    if num_layer is None:
        num_layer = len(model0.layers) + 1

    hard_match = [None] * (num_layer - 1)
    random_match = [None] * (num_layer - 1)

    corr_unaligned_mean = np.zeros(num_layer)
    corr_aligned_mean = np.zeros(num_layer)

    if return_corr is not None:
        corr_unaligned_returned = []
        corr_aligned_returned = []
    else:
        corr_unaligned_returned = []
        corr_aligned_returned = []

    acts_old0 = None
    acts_old1 = None
    for layer in range(num_layer):
        print('Layer %d' % layer)

        if not quad_assignment:
            if use_warmstart:
                corr, acts_old0, acts_old1 = cc.compute_corr(model0, model1, dataloader, layer, acts_old0, acts_old1,
                                                              idx_warmstart=layer - 1)
            else:
                corr, _, _ = cc.compute_corr(model0, model1, dataloader, layer, idx_warmstart=None, use_warmstart=False)
            if layer < num_layer - 1:
                hard_match[layer] = compute_alignment(corr)
                random_match[layer] = compute_alignment_random(corr, seed=layer)
                corr_aligned = corr[:, hard_match[layer][:, 1]]
            else:
                corr_aligned = corr
            corr_unaligned_mean[layer] = np.mean(np.diag(corr))
            corr_aligned_mean[layer] = np.mean(np.diag(corr_aligned))

            if return_corr is not None:
                if layer in return_corr:
                    corr_unaligned_returned.append(corr)
                    corr_aligned_returned.append(corr_aligned)
                if layer >= np.max(return_corr):
                    return hard_match, random_match, corr_unaligned_mean, corr_aligned_mean, \
                           corr_unaligned_returned, corr_aligned_returned

    return hard_match, random_match, corr_unaligned_mean, corr_aligned_mean


def compute_model_alignment_resnet(model0, model1, dataloader, return_corr=None):
    num_layer = 32

    hard_match = [None] * (num_layer - 1)
    hard_match_noid = [None] * (num_layer - 1)
    random_match = [None] * (num_layer - 1)

    corr_unaligned_mean = np.zeros(num_layer)
    corr_aligned_mean = np.zeros(num_layer)
    corr_aligned_noid_mean = np.zeros(num_layer)

    if return_corr is not None:
        corr_unaligned_returned = [None] * len(return_corr)
        corr_aligned_returned = [None] * len(return_corr)
        corr_aligned_noid_returned = [None] * len(return_corr)
    else:
        corr_unaligned_returned = None
        corr_aligned_returned = None
        corr_aligned_noid_returned = None

    print('Layer %d' % 0)
    _, corr, acts_old0, acts_old1 = cc.compute_corr_resnet(model0, model1, dataloader, 0)
    hard_match[0] = compute_alignment(corr)
    hard_match_noid[0] = hard_match[0]
    random_match[0] = compute_alignment_random(corr, seed=0)

    corr_aligned_noid = corr[:, hard_match[0][:, 1]]
    corr_unaligned_mean[0] = np.mean(np.diag(corr))
    corr_aligned_noid_mean[0] = np.mean(np.diag(corr_aligned_noid))
    corr_aligned_mean[0] = np.mean(np.diag(corr_aligned_noid))

    if return_corr is not None:
        if 0 in return_corr:
            corr_unaligned_returned[0] = corr
            corr_aligned_returned[0] = corr_aligned_noid
            corr_aligned_noid_returned[0] = corr_aligned_noid
        if 0 >= np.max(return_corr):
            return hard_match, hard_match_noid, random_match, corr_unaligned_mean, corr_aligned_mean, \
                   corr_aligned_noid_mean, corr_unaligned_returned, corr_aligned_returned, corr_aligned_noid_returned
    i = 1
    for block in model0.layers[1:]:
        update_list = []
        num_filter = block[0].conv1.out_channels
        corr = np.zeros([num_filter, num_filter])
        corr_list = []
        for _ in block:
            print('Layers %d, %d' % (2 * i - 1, 2 * i))
            update_list.append(2*i)

            corr_int, corr_temp, acts_old0, acts_old1 = cc.compute_corr_resnet(
                model0, model1, dataloader, i, acts_old0=acts_old0, acts_old1=acts_old1, idx_warmstart=i - 1)

            corr += corr_temp
            corr_list.append(corr_temp)

            hard_match_noid[2 * i - 1] = compute_alignment(corr_int)
            hard_match[2 * i - 1] = hard_match_noid[2 * i - 1]
            random_match[2 * i - 1] = compute_alignment_random(corr_int, seed=2 * i - 1)

            hard_match_noid[2 * i] = compute_alignment(corr_temp)
            random_match[2 * i] = compute_alignment_random(corr_temp, seed=2 * i)

            corr_unaligned_mean[2 * i - 1] = np.mean(np.diag(corr_int))
            corr_unaligned_mean[2 * i] = np.mean(np.diag(corr_temp))

            corr_aligned_int = corr_int[:, hard_match[2 * i - 1][:, 1]]
            corr_aligned_noid = corr_temp[:, hard_match_noid[2 * i][:, 1]]
            corr_aligned_noid_mean[2 * i - 1] = np.mean(np.diag(corr_aligned_int))
            corr_aligned_noid_mean[2 * i] = np.mean(np.diag(corr_aligned_noid))

            corr_aligned_mean[2 * i - 1] = corr_aligned_noid_mean[2 * i - 1]

            if return_corr is not None:
                if 2 * i - 1 in return_corr:
                    corr_idx = return_corr.index(2 * i - 1)
                    corr_unaligned_returned[corr_idx] = corr
                    corr_aligned_returned[corr_idx] = corr_aligned_int
                    corr_aligned_noid_returned[corr_idx] = corr_aligned_int
                if 2 * i in return_corr:
                    corr_idx = return_corr.index(2 * i)
                    corr_unaligned_returned[corr_idx] = corr
                    corr_aligned_noid_returned[corr_idx] = corr_aligned_noid
            i += 1
        hard_match_block = compute_alignment(corr)
        for j in update_list:
            hard_match[j] = hard_match_block

            corr_item = corr_list.pop(0)
            corr_aligned = corr_item[:, hard_match[j][:, 1]]
            corr_aligned_mean[j] = np.mean(np.diag(corr_aligned))

            if return_corr is not None:
                if j in return_corr:
                    corr_idx = return_corr.index(j)
                    corr_aligned_returned[corr_idx] = corr_aligned
                if j >= np.max(return_corr):
                    return hard_match, hard_match_noid, random_match, corr_unaligned_mean, corr_aligned_mean, \
                           corr_aligned_noid_mean, corr_unaligned_returned, corr_aligned_returned, \
                           corr_aligned_noid_returned

    print('Layer %d' % (2*i - 1))
    _, cross_corr_temp, _, _ = cc.compute_corr_resnet(model0, model1, dataloader, i, acts_old0, acts_old1,
                                                      idx_warmstart=i-1)
    corr_unaligned_mean[-1] = np.mean(np.diag(cross_corr_temp))
    corr_aligned_noid_mean[-1] = corr_unaligned_mean[-1]
    corr_aligned_mean[-1] = corr_unaligned_mean[-1]

    if return_corr is not None:
        if num_layer-1 in return_corr:
            corr_unaligned_returned[-1] = corr
            corr_aligned_returned[-1] = corr
            corr_aligned_noid_returned[-1] = corr
        if num_layer-1 >= np.max(return_corr):
            return hard_match, hard_match_noid, random_match, corr_unaligned_mean, corr_aligned_mean, \
                   corr_aligned_noid_mean, corr_unaligned_returned, corr_aligned_returned, corr_aligned_noid_returned

    return hard_match, hard_match_noid, random_match, corr_unaligned_mean, corr_aligned_mean, corr_aligned_noid_mean


def compute_model_alignment_w2_pre(model0, model1, dataloader, num_layer=None, return_corr=None, quad_assignment=False,
                                   use_warmstart=True, pre_act=True):
    if num_layer is None:
        num_layer = len(model0.layers) + 1

    hard_match = [None] * (num_layer - 1)
    random_match = [None] * (num_layer - 1)

    corr_unaligned_mean = np.zeros(num_layer)
    corr_aligned_mean = np.zeros(num_layer)

    if return_corr is not None:
        corr_unaligned_returned = []
        corr_aligned_returned = []
    else:
        corr_unaligned_returned = []
        corr_aligned_returned = []

    acts_old0 = None
    acts_old1 = None
    for layer in range(num_layer):
        print('Layer %d' % layer)

        if not quad_assignment:
            if use_warmstart:
                w2, acts_old0, acts_old1, corr, acts_old00, acts_old10 = \
                    cc.compute_w2(model0, model1, dataloader, layer, acts_old0, acts_old1, idx_warmstart=layer - 1,
                                  pre_act=pre_act)
            else:
                w2, _, _, corr, _, _ = cc.compute_w2(model0, model1, dataloader, layer, idx_warmstart=None,
                                                     use_warmstart=False, pre_act=pre_act)
            if layer < num_layer - 1:
                hard_match[layer] = compute_alignment(corr, neg=True)
                random_match[layer] = compute_alignment_random(corr, seed=layer)
                corr_aligned = corr[:, hard_match[layer][:, 1]]
                w2_aligned, _, _, _, _, _ = cc.compute_w2(model0, model1, dataloader, layer, acts_old00, acts_old10,
                                                          idx_warmstart=layer - 1, pre_act=pre_act,
                                                          P=hard_match[layer][:, 1])
            else:
                w2_aligned = w2
            corr_unaligned_mean[layer] = (np.sum(np.diag(w2)) / len(dataloader.dataset)) ** 0.5
            corr_aligned_mean[layer] = (np.sum(np.diag(w2_aligned)) / len(dataloader.dataset)) ** 0.5

            if return_corr is not None:
                if layer in return_corr:
                    corr_unaligned_returned.append(corr)
                    corr_aligned_returned.append(corr_aligned)
                if layer >= np.max(return_corr):
                    return hard_match, random_match, corr_unaligned_mean, corr_aligned_mean, \
                           corr_unaligned_returned, corr_aligned_returned

    return hard_match, random_match, corr_unaligned_mean, corr_aligned_mean


def align_models(model, matching, selected_layers=None):
    if selected_layers is None:
        selected_layers = np.arange(len(model.layers) + 1)
    model_new = copy.deepcopy(model)
    for i, block in enumerate(model_new.layers):
        if i in selected_layers:
            for layer in block.modules():
                align_weights_head(layer, matching[i])
                if i > 0:
                    align_weights_tail(layer, matching[i-1])
    if len(model.layers) in selected_layers:
        if hasattr(model_new, 'dense'):  # TinyTen classifier
            pad_int = int(model_new.dense.in_features / matching[-1].size)
            new_match = []
            for mat in matching[-1]:
                new_match += [mat * pad_int + m_idx for m_idx in range(pad_int)]
            align_weights_tail(model_new.dense, new_match)
        if hasattr(model_new, 'classifier'):
            align_weights_tail(model_new.classifier, matching[-1])
    return model_new


def align_models_resnet(model, matching, selected_blocks=None):
    if selected_blocks is None:
        selected_blocks = np.arange(len(model.layers) + 1)

    model_new = copy.deepcopy(model)
    matching_new = copy.deepcopy(matching)
    idx = 0

    if idx in selected_blocks:
        for layer in model_new.layers[idx]:
            align_weights_head(layer, matching_new[idx])

    for i, block in enumerate(model_new.layers[1:]):
        if i+1 in selected_blocks:
            for res_unit in block:
                idx += 1
                if res_unit.downsample is not None:
                    for layer in res_unit.downsample:
                        align_weights_head(layer, matching_new[2*idx])
                        align_weights_tail(layer, matching_new[2*idx-2])
                else:
                    matching_new[2*idx] = matching_new[2*idx-2]  # Alignment should be propogating through the res block

                for layer_name, layer_val in res_unit.named_modules():
                    if layer_name.endswith('1') and not layer_name.startswith('downsample'):
                        # These belong to the interior residual block layers
                        align_weights_head(layer_val, matching_new[2*idx - 1])
                        align_weights_tail(layer_val, matching_new[2*idx - 2])
                    if layer_name.endswith('2') and not layer_name.startswith('downsample'):
                        # These belong to the exterior residual block layers
                        align_weights_head(layer_val, matching_new[2*idx])
                        align_weights_tail(layer_val, matching_new[2*idx - 1])
    if len(model.layers) in selected_blocks:
        align_weights_tail(model_new.fc, matching_new[-1])
    return model_new, matching_new


def align_weights_head(layer, match):
    match = np.array(match, dtype=np.int) 
    if match.ndim == 1:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.data = layer.weight.data[match]
            if layer.bias is not None:
                layer.bias.data = layer.bias.data[match]
        if isinstance(layer, nn.BatchNorm2d):
            layer.weight.data = layer.weight.data[match]
            layer.bias.data = layer.bias.data[match]
            layer.running_mean = layer.running_mean[match]
            layer.running_var = layer.running_var[match]
    else:
        assert match.ndim == 2
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.data = torch.matmul(torch.tensor(match, device=layer.weight.device), layer.weight.data)
            if layer.bias is not None:
                layer.bias.data = torch.matmul(torch.tensor(match, device=layer.bias.device), layer.bias.data)
        if isinstance(layer, nn.BatchNorm2d):
            layer.weight.data = torch.matmul(torch.tensor(match, device=layer.weight.device), layer.weight.data)
            layer.bias.data = torch.matmul(torch.tensor(match, device=layer.bias.device), layer.bias.data)
            layer.running_var = torch.matmul(torch.tensor(match, device=layer.running_var.device), layer.running_var)
            layer.running_mean = torch.matmul(torch.tensor(match, device=layer.running_mean.device), layer.running_mean)


def align_weights_tail(layer, match):
    match = np.array(match, dtype=np.int) 
    if match.ndim == 1:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.data = layer.weight.data[:, match]
    else:
        assert match.ndim == 2
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            match_t = torch.tensor(match, device=layer.weight.device)
            layer.weight.data = torch.matmul(layer.weight.data, match_t.t())
