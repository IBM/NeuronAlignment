import torch
import numpy as np


def train_perm_orth(train_loader, model, optimizer, scheduler, criterion, regularizer=None, rho=1E-4, delta=0.5,
                    nu=1E-2, eps=1E-3, tau=1E-2, lagrange_pen=1E-2, perm_flag=True, t_step=40):
    if perm_flag:
        tau_min = 1E-24
        tau_max = 1E-1
        c = None
        lam_lm = []
        for p in optimizer.param_groups[0]['params']:
            lam_lm.append(torch.zeros_like(p))

        k_iter = 0
        ts = torch.empty(len(train_loader), device=model.device).uniform_(0.0, 1.0)
        with torch.no_grad():
            for p in optimizer.param_groups[0]['params']:
                p.data = torch.rand_like(p.data)
                p.data, _, _ = torch.svd(p.data)
        input_cml = []
        target_cml = []
        t_cml = []
        inner_iter = 0
        loss = 0.0
        loss_obj = 0.0
        for iter, (input, target) in enumerate(train_loader):
            t = ts[iter]
            input = input.to(model.device, non_blocking=False)
            target = target.to(model.device, non_blocking=False)
            output = model(input, perm_train=True, t=t)
            input_all = input
            target_all = target

            new_loss = criterion(output, target_all)
            loss_obj += new_loss

            #  This part is for the augmented Lagrangian method
            int_pen = integer_penalty(optimizer.param_groups[0]['params'], lam_lm, lagrange_pen)

            loss += new_loss + int_pen

            inner_iter += 1
            input_cml.append(input.clone())
            target_cml.append(target.clone())
            t_cml.append(t.clone())
            if inner_iter % t_step == 0:
                optimizer.zero_grad()
                loss.backward()
                grad_norm = 0.0
                violator = 0.0
                for p in optimizer.param_groups[0]['params']:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
                    violator += torch.sum((torch.matmul(p.data.t(), p.data) - torch.eye(p.data.shape[0],
                                                                                        device=p.device)) ** 2)
                grad_norm = grad_norm ** (1. / 2)

                if c is None:
                    c = loss.clone().item()
                    q_opt = 1
                loss_inner = loss.clone()
                print('Iteration: %03d, Loss %.2E, Objective %.2E, Negative Penalty: %.2E,'
                      'Grad Norm: %.2E, Ortho Violation: %.2E, tau: %.2E' %
                      (k_iter, loss_inner.item(), loss_obj.item(), int_pen.item(), grad_norm, violator.item(), tau))
                #  Compute F for defining Y function
                F_list = []
                with torch.no_grad():
                    for p in optimizer.param_groups[0]['params']:
                        f = torch.matmul(p.grad.data, p.t().data) - torch.matmul(p.data, p.grad.t().data)
                        F_list.append(f)
                    #  Store old parameters
                    params_old = [None] * len(optimizer.param_groups[0]['params'])
                    for idx, param in enumerate(optimizer.param_groups[0]['params']):
                        params_old[idx] = param.clone()
                    grads_old = [p.grad.data.clone() for p in optimizer.param_groups[0]['params']]

                    #  Compute the values of Y(tau) and Y'(tau), store them into the model
                    Y_t, Y_ft_prime = compute_ytau(tau, F_list, optimizer.param_groups[0]['params'])
                    for p, y_t in zip(optimizer.param_groups[0]['params'], Y_t):
                        p.data = y_t.clone()
                loss_inner = 0.0
                for t_2, input_2, target_2 in zip(t_cml, input_cml, target_cml):
                    output = model(input_2, perm_train=True, t=t_2)

                    loss_inner += criterion(output, target_2)
                    int_pen = integer_penalty(optimizer.param_groups[0]['params'], lam_lm, lagrange_pen)
                    loss_inner += int_pen

                optimizer.zero_grad()
                loss_inner.backward()
                grads_new = [p.grad.data.clone() for p in optimizer.param_groups[0]['params']]

                with torch.no_grad():
                    dF_dt = 0.0
                    for g_new, y_ft_p in zip(grads_new, Y_ft_prime):
                        df = g_new * (y_ft_p / torch.norm(y_ft_p.data))
                        df = torch.sum(df)
                        dF_dt += df.item()

                threshold_flag = True
                k_inner = 0
                while threshold_flag:
                    with torch.no_grad():
                        threshold = c + rho * tau * dF_dt
                    if loss_inner.item() >= threshold:
                        # Compute Y for smaller value of tau
                        with torch.no_grad():
                            tau *= delta
                            Y_t, Y_ft_prime = compute_ytau(tau, F_list, optimizer.param_groups[0]['params'])
                            for p, y_t in zip(optimizer.param_groups[0]['params'], Y_t):
                                p.data = y_t.clone()

                        loss_old = loss_inner.clone()
                        loss_inner = 0.0
                        for t_2, input_2, target_2 in zip(t_cml, input_cml, target_cml):
                            output = model(input_2, perm_train=True, t=t_2)
                            loss_inner += criterion(output, target_2)
                            int_pen = integer_penalty(optimizer.param_groups[0]['params'], lam_lm, lagrange_pen)
                            loss_inner += int_pen

                        optimizer.zero_grad()
                        loss_inner.backward()
                        grads_new = [p.grad.data.clone() for p in optimizer.param_groups[0]['params']]

                        k_inner += 1
                        if (loss_inner.item() - loss_old.item()) / (1 + loss_old.item()) < 1E-5:
                            threshold_flag = False
                    else:
                        threshold_flag = False

                with torch.no_grad():
                    c = (nu * q_opt * c + loss_inner.item())
                    q_opt = nu * q_opt + 1
                    c = c / q_opt

                    bb_num = 0.0
                    bb_denom = 0.0
                    yy_sum = 0.0
                    for p_old, g_old, p_new, g_new in zip(params_old, grads_old, optimizer.param_groups[0]['params'],
                                                          grads_new):
                        s_bb = p_new - p_old
                        y_bb = g_new - g_old
                        bb_num += torch.sum(s_bb ** 2)
                        bb_denom += torch.sum(s_bb * y_bb)
                        yy_sum += torch.sum(y_bb ** 2)
                    tau_bb = bb_num / torch.abs(bb_denom)
                    tau_bb = tau_bb.item()
                    tau_bb2 = torch.abs(bb_denom) / yy_sum
                    tau_bb2 = tau_bb2.item()
                    tau_bb = np.minimum(tau_bb, tau_bb2)
                    tau = np.minimum(tau_bb, tau_max)
                    tau = np.maximum(tau, tau_min)
                    lam_lm, lagrange_pen = integer_penalty_update(optimizer.param_groups[0]['params'], lam_lm,
                                                                  lagrange_pen)

                loss_inner = 0.0
                for t_2, input_2, target_2 in zip(t_cml, input_cml, target_cml):
                    output = model(input_2, perm_train=True, t=t_2)
                    loss_obj = criterion(output, target_2)
                    int_pen = integer_penalty(optimizer.param_groups[0]['params'], lam_lm, lagrange_pen)
                    loss_inner += loss_obj + int_pen

                optimizer.zero_grad()
                loss_inner.backward()
                grads_new = [p.grad.data.clone() for p in optimizer.param_groups[0]['params']]

                grad_norm = 0.0
                for g_new in grads_new:
                    gn = g_new.norm(2)
                    grad_norm += gn.item() ** 2
                grad_norm = grad_norm ** (1. / 2)

                k_iter += 1
                input_cml = []
                target_cml = []
                t_cml = []
                loss = 0.0
                loss_obj = 0.0

    model.train()
    loss_sum = 0.0
    correct = 0.0

    change_P = np.nan

    params_before = [None] * len(optimizer.param_groups[0]['params'])
    if nu is not None:
        for idx, param in enumerate(optimizer.param_groups[0]['params']):
            params_before[idx] = param.clone().detach()

        optimizer.step()

        lr = scheduler.get_lr()[0]
        with torch.no_grad():
            for param, param_o in zip(optimizer.param_groups[0]['params'], params_old):
                param.data = 1 / (1 + lr / nu) * (param + lr / nu * param_o)

            output = model(input_all, perm_train=True)
            loss = criterion(output, target_all)
            if regularizer is not None:
                loss += regularizer(model)
            loss_sum += loss.item() * input.size(0)

        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target_all.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
        'change_perm': change_P
    }


def hard_int_penalty(p_list, pen=1E1):
    pen_loss = 0.0
    for p in p_list:
        p_mask = p.data * (p.data <= 0)
        pen_loss += pen * torch.sum(p_mask ** 2)
    return pen_loss


def integer_penalty(p_list, lam_list, mu):
    pen_loss = 0.0
    for p, lam in zip(p_list, lam_list):
        mask = (p - lam / mu) <= 0
        mask_alt = (p - lam / mu) > 0
        p_l = torch.sum((- lam * p + 0.5 * mu * (p ** 2)) * mask)
        p_l += torch.sum((-1/(2 * mu) * lam ** 2) * mask_alt)
        pen_loss += p_l
    return pen_loss


def integer_penalty_update(p_list, lam_list, mu):
    new_lam_list = []
    with torch.no_grad():
        for p, lam in zip(p_list, lam_list):
            upd = lam - mu * p
            new_lam_list.append(upd * (upd > 0))
        new_mu = mu * 1.01
    return new_lam_list, new_mu


def compute_ytau(tau, f_list, p_list):
    y_tau = []
    y_tau_prime = []
    for p, f in zip(p_list, f_list):
        eye = torch.eye(f.shape[0], device=f.device)
        qmat_inv = torch.inverse(eye + tau / 2 * f)
        y_ft = torch.matmul(qmat_inv, eye - tau / 2 * f)
        y_ft = torch.matmul(y_ft, p)
        y_ft_prime = - torch.matmul(qmat_inv, f)
        y_ft_prime = torch.matmul(y_ft_prime, (p + y_ft) / 2)

        y_tau.append(y_ft.clone())
        y_tau_prime.append(y_ft_prime.clone())
    return y_tau, y_tau_prime