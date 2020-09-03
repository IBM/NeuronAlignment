import numpy as np
import utils


def get_acts(model, dataloader, idx_layer, acts_old=None, idx_warmstart=None, use_warmstart=True, pre_act=False):
    act, acts = utils.get_activations(dataloader, model, idx_layer, acts_old=acts_old, idx_warmstart=idx_warmstart,
                                      use_warmstart=use_warmstart, pre_act=pre_act)
    return act, acts


def get_zscore(model, dataloader, idx_layer, acts_old=None, idx_warmstart=None, use_warmstart=True):
    eps = 1E-5
    act, acts = utils.get_activations(dataloader, model, idx_layer, acts_old=acts_old, idx_warmstart=idx_warmstart,
                                      use_warmstart=use_warmstart)
    act_mu = np.expand_dims(np.mean(act, axis=1), -1)
    act_sigma = np.expand_dims(np.std(act, axis=1), -1) + eps  # Avoid divide by zero
    z_score = (act - act_mu) / act_sigma
    return z_score, acts


def get_zscore_resnet(model, dataloader, idx_layer, acts_old=None, idx_warmstart=None, use_warmstart=True):
    z_score_int = None
    eps = 1E-5
    act_int, act, acts = utils.get_activations_resnet(dataloader, model, idx_layer, acts_old=acts_old,
                                                      idx_warmstart=idx_warmstart, use_warmstart=use_warmstart)
    if act_int is not None:
        act_int_mu = np.expand_dims(np.mean(act_int, axis=1), -1)
        act_int_sigma = np.expand_dims(np.std(act_int, axis=1), -1) + eps  # Avoid divide by zero
        z_score_int = (act_int - act_int_mu) / act_int_sigma

    act_mu = np.expand_dims(np.mean(act, axis=1), -1)
    act_sigma = np.expand_dims(np.std(act, axis=1), -1) + eps  # Avoid divide by zero
    z_score = (act - act_mu) / act_sigma
    return z_score_int, z_score, acts


def compute_w2(model0, model1, dataloader, idx_layer, acts_old0=None, acts_old1=None, idx_warmstart=None,
               use_warmstart=True, pre_act=True, P=None):
    act0, acts0 = get_acts(model0, dataloader, idx_layer, acts_old=acts_old0, idx_warmstart=idx_warmstart,
                           use_warmstart=use_warmstart, pre_act=pre_act)
    act1, acts1 = get_acts(model1, dataloader, idx_layer, acts_old=acts_old1, idx_warmstart=idx_warmstart,
                           use_warmstart=use_warmstart, pre_act=pre_act)
    # ||act0 - P act1||^2 = trace(act0^t act0 - 2 act0^t P act1 + act1^t act1)
    if P is None:
        diff = act0 - act1
    else:
        diff = act0 - act1[P]
    w2 = np.matmul(diff, diff.transpose())
    cov = np.matmul(act0, act1.transpose())
    return w2, acts0, acts1, cov, acts_old0, acts_old1


def compute_corr(model0, model1, dataloader, idx_layer, acts_old0=None, acts_old1=None, idx_warmstart=None,
                 use_warmstart=True):
    z_score0, acts0 = get_zscore(model0, dataloader, idx_layer, acts_old=acts_old0, idx_warmstart=idx_warmstart,
                                 use_warmstart=use_warmstart)
    z_score1, acts1 = get_zscore(model1, dataloader, idx_layer, acts_old=acts_old1, idx_warmstart=idx_warmstart,
                                 use_warmstart=use_warmstart)
    corr = np.matmul(z_score0, z_score1.transpose()) / z_score0.shape[1]
    return corr, acts0, acts1


def compute_corr_within(model, dataloader, idx_layer, acts_old=None, idx_warmstart=None):
    z_score, acts = get_zscore(model, dataloader, idx_layer, acts_old=acts_old, idx_warmstart=idx_warmstart)
    corr = np.matmul(z_score, z_score.transpose()) / z_score.shape[1]
    return corr, acts


def compute_corr_resnet(model0, model1, dataloader, idx_layer, acts_old0=None, acts_old1=None, idx_warmstart=None,
                        use_warmstart=True):
    corr_int = None
    z_score0_int, z_score0, acts0 = get_zscore_resnet(model0, dataloader, idx_layer, acts_old=acts_old0,
                                                      idx_warmstart=idx_warmstart, use_warmstart=use_warmstart)
    z_score1_int, z_score1, acts1 = get_zscore_resnet(model1, dataloader, idx_layer, acts_old=acts_old1,
                                                      idx_warmstart=idx_warmstart, use_warmstart=use_warmstart)

    if z_score0_int is not None:
        corr_int = np.matmul(z_score0_int, z_score1_int.transpose()) / z_score0.shape[1]
    corr = np.matmul(z_score0, z_score1.transpose()) / z_score0.shape[1]
    return corr_int, corr, acts0, acts1


def compute_corr_resnet_within(model, dataloader, idx_layer, acts_old=None, idx_warmstart=None):
    corr_int = None
    z_score_int, z_score, acts = get_zscore_resnet(model, dataloader, idx_layer, acts_old=acts_old,
                                                   idx_warmstart=idx_warmstart)
    if z_score_int is not None:
        corr_int = np.matmul(z_score_int, z_score_int.transpose()) / z_score.shape[1]
    corr = np.matmul(z_score, z_score.transpose()) / z_score.shape[1]
    return corr_int, corr, acts
