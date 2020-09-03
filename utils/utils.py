import numpy as np
import os
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from models import curves, curves_pam
from utils import attack_ensemble, attack_curve
from tqdm import tqdm
from utils.birkhoff import birkhoff_von_neumann_decomposition


def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sum(p ** 2)
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, adversarial_flag=False, tqdm_summary=False, amp_flag=False,
          **kwargs):
    loss_sum = 0.0
    correct = 0.0

    model.train()
    if tqdm_summary:
        iterator = enumerate(tqdm(train_loader, desc='Training batch:', leave=False))
    else:
        iterator = enumerate(train_loader)
    for iter, (input, target) in iterator:
        input = input.to(model.device, non_blocking=False)
        target = target.to(model.device, non_blocking=False)
        if adversarial_flag:
            kwargs = dict(kwargs, targets=target)
        output = model(input, **kwargs)
        if adversarial_flag:
            output = output[0]
        loss = criterion(output, target)
        if regularizer is not None:
            if adversarial_flag:
                loss += regularizer(model.basic_net)
            else:
                loss += regularizer(model)

        optimizer.zero_grad()

        loss.backward()
        # if amp_flag:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss.backward()

        optimizer.step()
        loss_sum += loss.item() * input.size(0)
        if isinstance(output, tuple):
            pred = output[0].data.argmax(1, keepdim=True)
        else:
            pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()
    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def train_perm(train_loader, model, optimizer, scheduler, criterion, params_old, regularizer=None, proj_freq=16, nu=-2,
               proj_flag=False, t_minibatch_sz=4, pen=1E-1, pen_flag=False, lp_pen=None, tqdm_summary=False):
    loss_sum = 0.0
    correct = 0.0

    change_P = np.nan

    params_before = [None] * len(optimizer.param_groups[0]['params'])
    if pen_flag:
        for idx, param in enumerate(optimizer.param_groups[0]['params']):
            params_before[idx] = param.detach().clone()
    if proj_flag:
        for idx, param in enumerate(optimizer.param_groups[0]['params']):
            params_before[idx] = param.detach().clone()

    if tqdm_summary:
        iterator = enumerate(tqdm(train_loader, desc='Training batch:', leave=False))
    else:
        iterator = enumerate(train_loader)

    model.train()
    for iter, (input, target) in iterator:
        loss = None
        if proj_flag:
            t_num_batch = t_minibatch_sz
        else:
            t_num_batch = 1

        input = input.to(model.device, non_blocking=False)
        target = target.to(model.device, non_blocking=False)

        for i in range(t_num_batch):
            if loss is None:
                output = model(input, perm_train=True)
                input_all = input
                target_all = target
            else:
                output_sub = model(input, perm_train=True)
                output = torch.cat((output, output_sub), dim=0)
                input_all = torch.cat((input_all, input), dim=0)
                target_all = torch.cat((target_all, target), dim=0)
        loss = criterion(output, target_all)
        if regularizer is not None:
            loss_reg = regularizer(model)
            loss += loss_reg

        lp_norm = 0.0
        if lp_pen is not None:
            for param in optimizer.param_groups[0]['params']:
                lp_norm += torch.sum((param + 1E-2) ** 0.75)
            loss += lp_pen * lp_norm

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        lr = scheduler.get_lr()[0]
        with torch.no_grad():

            for param, param_o in zip(optimizer.param_groups[0]['params'], params_old):
                param.data = 1 / (1 + lr / nu) * (param + lr / nu * param_o)
            if proj_flag or pen_flag:
                project_doubly_stoch(model)
            output = model(input_all, perm_train=True)
            loss = criterion(output, target_all)
            if regularizer is not None:
                loss += regularizer(model)

            lp_norm = 0.0
            if lp_pen is not None:
                for param in optimizer.param_groups[0]['params']:
                    lp_norm += torch.sum((param + 1E-2) ** 0.75)
                loss += lp_pen * lp_norm

            loss_sum += loss.item() * input.size(0)

        if isinstance(output, tuple):
            pred = output[0].data.argmax(1, keepdim=True)
        else:
            pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target_all.data.view_as(pred)).sum().item()

    if proj_flag:
        project_permutations(model)

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset)
    }


def project_permutations(model):
    with torch.no_grad():
        for perm in model.permutations.parameters():
            row_ind, col_ind = linear_sum_assignment(-perm.cpu().numpy())
            perm_new = torch.zeros(perm.shape)
            perm_new[row_ind, col_ind] = 1
            perm_new = perm_new.to(model.device)
            perm.data = perm_new


def sample_permutation(model, optimizer, train_loader, test_loader, criterion, old_perm, k=32):
    opt_loss = np.inf
    doubly_stoch = []
    new_perm = []

    perms_bvn = []
    cdf_bvn = []
    for i_bvn, param in enumerate(optimizer.param_groups[0]['params']):
        doubly_stoch.append(param.clone())
        new_perm.append(param.clone())

        ds_try = param.clone().detach().cpu().numpy()
        bvn_result = birkhoff_von_neumann_decomposition(ds_try)
        bvn_coeffs = np.array([c for c, _ in bvn_result])
        bvn_perms = np.array([p for _, p in bvn_result])

        bvn_idx = np.argsort(bvn_coeffs)
        bvn_coeffs = bvn_coeffs[bvn_idx]
        bvn_perms = bvn_perms[bvn_idx]

        bvn_coeffs = bvn_coeffs[-11:]
        bvn_perms = bvn_perms[-11:]
        bvn_cdf = np.cumsum(bvn_coeffs)
        bvn_cdf = bvn_cdf / bvn_cdf[-1]

        perms_bvn.append(bvn_perms)
        cdf_bvn.append(bvn_cdf)
        print('Birkhoff von Neumann Decomposition computed for %02d layer permutation' % i_bvn)
    with torch.no_grad():
        project_permutations(model)
        model.train()
        test_dict = test_perm(test_loader, model, criterion, bn_eval=False, train_loader=train_loader, samp_t=True)
        opt_loss = test_dict['nll']
        for param, param_opt in zip(optimizer.param_groups[0]['params'], new_perm):
            param_opt.data = param.clone()
        print('Permutation Projection Baseline; Optimal Loss: %.2E, Loss: %.2E, Accuracy: %.2f'
              % (opt_loss, test_dict['nll'], test_dict['accuracy']))
        for param, param_old in zip(optimizer.param_groups[0]['params'], old_perm):
            param.data = param_old.data
        test_dict = test_perm(test_loader, model, criterion, bn_eval=False, train_loader=train_loader, samp_t=True)
        old_loss = test_dict['nll']
        if old_loss <= opt_loss:
            for param, param_opt in zip(optimizer.param_groups[0]['params'], new_perm):
                param_opt.data = param.clone()
        for k_i in range(k):
            prob_samples = np.random.uniform(0, 1, len(doubly_stoch))
            for param_ds, param_model, c_bvn, p_bvn, p_s in \
                    zip(doubly_stoch, optimizer.param_groups[0]['params'], cdf_bvn, perms_bvn, prob_samples):
                p_idx = np.argmax(c_bvn >= p_s)
                p = p_bvn[p_idx]
                perm = torch.tensor(p, device=param_model.device, dtype=torch.float32)
                param_model.data = perm.clone()
            model.train()
            test_dict = test_perm(test_loader, model, criterion, bn_eval=False, train_loader=train_loader, samp_t=True)
            if test_dict['nll'] < opt_loss:
                opt_loss = test_dict['nll']
                for param, param_opt in zip(optimizer.param_groups[0]['params'], new_perm):
                    param_opt.data = param.clone()
            print('Permutation Sample: %02d, Optimal Loss: %.2E, Loss: %.2E, Accuracy: %.2f'
                  % (k_i, opt_loss, test_dict['nll'], test_dict['accuracy']))
        for param, param_opt in zip(optimizer.param_groups[0]['params'], new_perm):
            param.data = param_opt.clone()


def project_doubly_stoch(model):
    with torch.no_grad():
        for perm in model.permutations.parameters():
            n = perm.shape[0]
            ones_col = torch.ones([n, 1], device=perm.device)
            ones_row = torch.ones([1, n], device=perm.device)
            for _ in range(20):
                perm.data = F.relu(perm)
                proxy = -1 / n * torch.matmul(torch.sum(perm, dim=1, keepdim=True) - 1, ones_row)
                proxy -= 1 / n * torch.matmul(ones_col, torch.sum(perm, dim=0, keepdim=True) - 1)
                proxy += 1 / n ** 2 * (torch.sum(perm) - n)
                perm.data += proxy


def test(test_loader, model, criterion, regularizer=None, train_loader=None, adversarial_flag=False, tqdm_summary=False,
         bn_eval=True, samp_t=False, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0
    loss_time = None
    acc_time = None
    if samp_t and (hasattr(model, 'fix_points') or isinstance(model, attack_curve.AttackPGD)):
        if check_bn(model):
            num_batch = sum(1 for _ in test_loader)
            t_vec = np.linspace(0, 1, num_batch)

            loss_time = np.zeros(num_batch)
            acc_time = np.zeros(num_batch)
    if bn_eval:
        model.eval()
    else:
        model.train() 
    if tqdm_summary:
        iterator = tqdm(test_loader, desc='Testing batch', leave=False)
    else:
        iterator = test_loader
    with torch.no_grad():
        time_idx = 0
        for input, target in iterator:
            if samp_t:
                t = torch.tensor(t_vec[0])
                t_vec = t_vec[1:]
                t = t.to(model.device)
                if bn_eval: 
                    update_bn(train_loader, model, t=t)
                kwargs['t'] = t
            input = input.to(model.device, non_blocking=False)
            target = target.to(model.device, non_blocking=False)
            if adversarial_flag:
                kwargs = dict(kwargs, targets=target)

            output = model(input, **kwargs)
            if adversarial_flag:
                output = output[0]
            nll = criterion(output, target)
            loss = nll.clone()
            if regularizer is not None:
                if adversarial_flag:
                    loss += regularizer(model.basic_net)
                else:
                    loss += regularizer(model)

            nll_sum += nll.item() * input.size(0)
            loss_sum += loss.item() * input.size(0)
            if isinstance(output, tuple):
                pred = output[0].data.argmax(1, keepdim=True)
            else:
                pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

            if samp_t:
                loss_time[time_idx] = nll.item()
                acc_time[time_idx] = pred.eq(target.data.view_as(pred)).sum().item()
            time_idx += 1
    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
        'loss_time': loss_time,
        'acc_time': acc_time
    }


def test_perm(test_loader, model, criterion, regularizer=None, train_loader=None, bn_eval=True, samp_t=False, **kwargs):
    num_batch = len(test_loader)
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0
    loss_time = None
    acc_time = None
    if hasattr(model, 'fix_points') and samp_t:
        if check_bn(model):
            t_vec = torch.linspace(0, 1, num_batch, device=model.device)
            loss_time = np.zeros(num_batch)
            acc_time = np.zeros(num_batch)
    if bn_eval:
        model.eval()
    else:
        model.train()
    with torch.no_grad():
        time_idx = 0
        for iter, (input, target) in enumerate(test_loader):
            if samp_t:
                if bn_eval:
                    update_bn(train_loader, model, t=t_vec[iter], perm_train=True)
                kwargs['t'] = t_vec[iter]
            
            input = input.to(model.device, non_blocking=False)
            target = target.to(model.device, non_blocking=False)

            output = model(input, perm_train=True, **kwargs)
            nll = criterion(output, target)
            loss = nll.clone()
            if regularizer is not None:
                loss += regularizer(model)

            nll_sum += nll.item() * input.size(0)
            loss_sum += loss.item() * input.size(0)
            if isinstance(output, tuple):
                pred = output[0].data.argmax(1, keepdim=True)
            else:
                pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

            if samp_t:
                loss_time[iter] = nll.item()
                acc_time[iter] = pred.eq(target.data.view_as(pred)).sum().item()
    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
        'loss_time': loss_time,
        'acc_time': acc_time
    }


def predictions(test_loader, model, adv_flag=False, **kwargs):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for input, target in test_loader:
            input = input.to(model.device, non_blocking=False)
            target = target.to(model.device) 
            if adv_flag:
                kwargs['targets'] = target 
            output = model(input, **kwargs)
            if adv_flag:
                output = output[0] 
            probs = F.softmax(output, dim=1)
            preds.append(probs.cpu().data.numpy())
            targets.append(target.cpu().numpy())
    return np.vstack(preds), np.concatenate(targets)


def predictions_ensemble_adversarial(test_loader, ensemble, config, attack_flag=False, **kwargs):
    for model in ensemble:
        model.eval()
    preds = []
    targets = []

    net = attack_ensemble.AttackPGD(ensemble, config)
    with torch.no_grad():
        for _, (input, target) in enumerate(tqdm(test_loader, desc='Batch')):
            probs = None

            if attack_flag:
                input = input.to(net.device)
                target = target.to(net.device) 
                input = net(input, target)[1]
            for idx, model in enumerate(ensemble):
                input = input.to(model.device)
                target = target.to(model.device)
                output = model(input)
                if probs is None:
                    probs = F.softmax(output, dim=1).cpu()
                else:
                    probs += F.softmax(output, dim=1).cpu()
            preds.append(probs.cpu().data.numpy())
            targets.append(target.cpu().numpy())
    return np.vstack(preds), np.concatenate(targets)


def get_activations(test_loader, model, idx_layer, acts_old=None, idx_warmstart=None, grad_enabled=False, review=True,
                    use_warmstart=True, pre_act=False):
    acts = []
    model.eval()
    if acts_old is None or not use_warmstart:
        with torch.set_grad_enabled(grad_enabled):
            for input, _ in test_loader:
                input = input.to(model.device, non_blocking=False)
                if pre_act:
                    act_new = model.get_activation(input, idx_act=idx_layer, pre_act=pre_act)
                else:
                    act_new = model.get_activation(input, idx_act=idx_layer)
                if act_new.dim() == 4 and act_new.shape[-1] > 32:
                    act_new = act_new[:, :, ::2, ::2] 
                acts += [act_new]
    
    else:
        with torch.set_grad_enabled(grad_enabled):
            for act_old in acts_old:
                act_old = act_old.to(model.device, non_blocking=False)
                if pre_act:
                    act_new = model.get_activation(act_old, idx_act=idx_layer, idx_warmstart=idx_warmstart,
                                                   pre_act=pre_act)
                else:
                    act_new = model.get_activation(act_old, idx_act=idx_layer, idx_warmstart=idx_warmstart)
                if act_new.dim() == 4 and act_new.shape[-1] > 32:
                    act_new = act_new[:, :, ::2, ::2] 
                acts += [act_new]
    act = torch.cat([torch.transpose(i, 0, 1) for i in acts], dim=1)
    if review:
        act = torch.flatten(act, start_dim=1)
    act = act.cpu().numpy()
    return act, acts


def get_activations_resnet(test_loader, model, idx_layer, acts_old=None, idx_warmstart=None, grad_enabled=False,
                           review=True, use_warmstart=True):
    acts = []
    acts_int = []
    model.eval()
    if acts_old is None or not use_warmstart:
        with torch.set_grad_enabled(grad_enabled):
            for input, _ in test_loader:
                input = input.to(model.device, non_blocking=False)
                acts_int_item, acts_item = model.get_activation(input, idx_act=idx_layer)
                if acts_int_item is not None:    
                    if acts_int_item.dim() == 4 and acts_int_item.shape[-1] > 32:
                        acts_int_item = acts_int_item[:, :, ::2, ::2]
                if acts_item is not None: 
                    if acts_item.dim() == 4 and acts_item.shape[-1] > 32:
                        acts_item = acts_item[:, :, ::2, ::2] 
                acts_int.append(acts_int_item)
                acts.append(acts_item)
    else:
        with torch.set_grad_enabled(grad_enabled):
            for act_old in acts_old:
                act_old = act_old.to(model.device, non_blocking=False)
                acts_int_item, acts_item = model.get_activation(act_old, idx_act=idx_layer, idx_warmstart=idx_warmstart)
                acts_int.append(acts_int_item)
                acts.append(acts_item)

    if acts_int_item is not None:
        act_int = torch.cat([torch.transpose(i,0,1) for i in acts_int], dim=1)
        if review:
            act_int = act_int.view(act_int.shape[0], -1)
        act_int = act_int.cpu().numpy()
    else:
        act_int = None
    act = torch.cat([torch.transpose(i,0,1) for i in acts], dim=1)
    if review:
        act = act.view(act.shape[0], -1)
    act = act.cpu().numpy()
    return act_int, act, acts


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm) or \
           issubclass(module.__class__, curves_pam._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def _set_momenta_val(module, momentum):
    if isbatchnorm(module):
        module.momentum = momentum


def reset_bn_set_momenta(module, momentum):
    if isbatchnorm(module):
        module.reset_running_stats()
        module.momentum = momentum


def update_bn(loader, model, **kwargs):
    model.train()
    with torch.no_grad():
        momenta = {}
        model.apply(reset_bn)
        model.apply(lambda module: _get_momenta(module, momenta))
        num_samples = 0
        for input, target in loader:
            input = input.to(model.device, non_blocking=False)
             
            batch_size = input.data.size(0)

            momentum = batch_size / (num_samples + batch_size)
            for module in momenta.keys():
                module.momentum = momentum
            if not isinstance(model, attack_curve.AttackPGD): 
                model(input, **kwargs)
            else:
                target = target.to(model.device, non_blocking=False)
                model(input, target, **kwargs)  
            num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))
    model.eval()


def update_momenta(model, momenta):
    model.apply(lambda module: _set_momenta_val(module, momenta))


def update_bn_writeout(loader, model, wo_running_mean=None, wo_running_var=None, **kwargs):
    outer_iter = 1
    model.train()
    with torch.no_grad():
        if wo_running_mean is None:
            momenta = {}
            model.apply(reset_bn)
            model.apply(lambda module: _get_momenta(module, momenta))
            num_samples = 0
            for _ in range(outer_iter):
                for input, _ in loader:
                    input = input.to(model.device, non_blocking=False)
                    batch_size = input.data.size(0)

                    momentum = batch_size / (num_samples + batch_size)
                    for module in momenta.keys():
                        module.momentum = momentum
                    model(input, **kwargs)
                    num_samples += batch_size

            model.apply(lambda module: _set_momenta(module, momenta))

            wo_running_mean_new = {}
            wo_running_var_new = {}
            for name, bn_layer in model.named_modules():
                if isbatchnorm(bn_layer):
                    wo_running_mean_new[name] = bn_layer.running_mean.data.clone()
                    wo_running_var_new[name] = bn_layer.running_var.data.clone()

        else:
            wo_running_mean_new = {}
            wo_running_var_new = {}
            for name, bn_layer in model.named_modules():
                if isbatchnorm(bn_layer):
                    bn_layer.running_mean = wo_running_mean[name].clone()
                    bn_layer.running_var = wo_running_var[name].clone()
    model.eval()
    return wo_running_mean_new, wo_running_var_new


def create_model_ortho(state_0, state_1, state_2, architecture, num_classes, device, loader):
    state_list = list(state_0.items())
    ortho_dict1 = copy.deepcopy(state_1)
    ortho_dict2 = copy.deepcopy(state_0)

    xy = 0
    xx = 0
    yy = 0
    for i in range(len(state_0)):
        name = state_list[i][0]
        if name.endswith('weight') or name.endswith('bias'):
            x = state_1[name] - state_0[name]
            y = state_2[name] - state_0[name]
            xy += torch.sum(x * y)
            xx += torch.sum(x * x)
            yy += torch.sum(y * y)
    proj_x = xy / torch.sqrt(xx)

    x_len = torch.max(proj_x, torch.sqrt(xx))
    y_len = torch.sqrt(yy - proj_x**2)

    for i in range(len(state_0)):
        name = state_list[i][0]
        if name.endswith('weight') or name.endswith('bias'):
            x = state_1[name] - state_0[name]
            y = state_2[name] - state_0[name]
            if proj_x > torch.sqrt(xx):
                ortho_dict1[name] = state_0[name] + x * (proj_x / torch.sqrt(xx))
            y_new = y - proj_x * (x / torch.sqrt(xx))
            ortho_dict2[name] += y_new

    model_ortho1 = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
    model_ortho1.load_state_dict(ortho_dict1)
    model_ortho1.to(device)
    if proj_x > torch.sqrt(xx):
        update_bn(loader, model_ortho1)

    model_ortho2 = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
    model_ortho2.load_state_dict(ortho_dict2)
    model_ortho2.to(device)
    update_bn(loader, model_ortho2)
    sqrt_xx = torch.sqrt(xx)
    model_1_coords = np.array([sqrt_xx.cpu().numpy(), 0])
    model_2_coords = np.array([proj_x.cpu().numpy(), y_len.cpu().numpy()])
    return model_ortho1, model_ortho2, x_len.cpu().numpy(), y_len.cpu().numpy(), model_1_coords, model_2_coords


def eval_plane(states, architecture, loaders, num_classes, num_points, device,
               t_min=[-0.05, -0.05], t_max=[1.05, 1.05], googlenet_flag=False):
    model_0 = architecture.base(num_classes=num_classes, device=device, **architecture.kwargs)
    model_0.load_state_dict(states[0])

    model_1, model_2, x_len, y_len, model1_coords, model2_coords = \
        create_model_ortho(states[0], states[1], states[2], architecture, num_classes, device, loaders['train'])
    print('Model space orthogonalized')

    model = architecture.curve(num_classes=num_classes, device=device, fix_points=[True] * 3, **architecture.kwargs)
    model.import_base_parameters(model_0, 0)
    model.import_base_parameters(model_1, 1)
    model.import_base_parameters(model_2, 2)
    model.to(device)
    if googlenet_flag:
        criterion = googlenet_criterion
    else:
        criterion = nn.CrossEntropyLoss()

    t1 = np.linspace(t_min[0], t_max[0], num_points)
    t2 = np.linspace(t_min[1], t_max[1], num_points)

    has_bn = check_bn(model_0)
    loss = np.zeros([num_points, num_points])
    acc = np.zeros([num_points, num_points])
    t = torch.tensor([0, 0, 0], dtype=torch.float32, device=device, requires_grad=False)
    for i, x in enumerate(t1):
        for j, y in enumerate(t2):
            time_ep = time.time()
            z = 1 - x - y
            t.copy_(torch.from_numpy(np.array([z, x, y])))
            if has_bn:
                update_bn(loaders['train'], model, coeffs_t=t)

            te_res = test(loaders['test'], model, criterion, coeffs_t=t)
            loss[i, j] = te_res['nll']
            acc[i, j] = te_res['accuracy']
            time_ep = time.time() - time_ep
            print('Point (%.3f, %.3f), Test Loss: %.2E, Test Accuracy: %.2f, Time Elapsed: %.2fs' %
                  (x, y, te_res['nll'], te_res['accuracy'], time_ep))
    return t1*x_len, t2*y_len, loss, acc, model1_coords, model2_coords


def googlenet_criterion(output, target):
    ce_loss = nn.CrossEntropyLoss()
    if isinstance(output, tuple):
        loss = ce_loss(output[0], target)
        if output[1] is not None:
            loss += 0.3 * ce_loss(output[1], target)
            loss += 0.3 * ce_loss(output[2], target)
    else:
        loss = ce_loss(output, target)
    return loss


def mse_loss_onehot(output, target):
    mse_loss = nn.MSELoss()
    target = nn.functional.one_hot(target.long(), num_classes=10).float()
    loss = mse_loss(output, target)
    return loss
