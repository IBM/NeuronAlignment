import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
from scipy.special import binom
import copy
import utils
import time


class Bezier(Module):
    def __init__(self, num_bends):
        super(Bezier, self).__init__()
        self.register_buffer(
            'binom',
            torch.Tensor(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32))
        )
        self.register_buffer('range', torch.arange(0, float(num_bends)))
        self.register_buffer('rev_range', torch.arange(float(num_bends - 1), -1, -1))

    def forward(self, t):
        return self.binom * \
               torch.pow(t, self.range) * \
               torch.pow((1.0 - t), self.rev_range)


class PolyChain(Module):
    def __init__(self, num_bends):
        super(PolyChain, self).__init__()
        self.num_bends = num_bends
        self.register_buffer('range', torch.arange(0, float(num_bends)))

    def forward(self, t):
        t_n = t * (self.num_bends - 1)
        return torch.max(self.range.new([0.0]), 1.0 - torch.abs(t_n - self.range))


class CurveModule(Module):
    def __init__(self, fix_points, parameter_names=()):
        super(CurveModule, self).__init__()
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.parameter_names = parameter_names
        self.l2 = 0.0

    def compute_weights_t(self, coeffs_t, orient=None):
        w_t = [None] * len(self.parameter_names)
        self.l2 = 0.0
        for i, parameter_name in enumerate(self.parameter_names):
            for j, coeff in enumerate(coeffs_t):
                parameter = getattr(self, '%s_%d' % (parameter_name, j))
                if parameter is not None:
                    if j + 1 == len(coeffs_t) and orient is not None:
                        parameter_permuted = self.weight_permutation_perm_train(parameter.clone(), orient[0], orient[1])
                    elif j + 2 == len(coeffs_t) and orient is not None:
                        parm_beg = getattr(self, '%s_%d' % (parameter_name, 0))
                        parm_end = getattr(self, '%s_%d' % (parameter_name, len(coeffs_t) - 1))
                        parm_perm_end = self.weight_permutation_perm_train(parm_end.clone(), orient[0], orient[1])
                        parameter_permuted = 0.5 * parm_beg + 0.5 * parm_perm_end + parameter
                    else:
                        parameter_permuted = parameter
                    if w_t[i] is None:
                        w_t[i] = parameter_permuted * coeff
                    else:
                        w_t[i] += parameter_permuted * coeff
            if w_t[i] is not None and self.training:
                self.l2 += torch.sum(w_t[i] ** 2)
        return w_t

    def weight_permutation_perm_train(self, param, perm_in, perm_out):
        param_ndim = len(param.shape)
        if param_ndim == 1:
            if perm_out is not None:    
                param = torch.matmul(perm_out, param)
        elif param_ndim == 2:
            if perm_in is not None:
                if param.shape[1] == perm_in.shape[0]:
                    param = torch.matmul(param, perm_in.t())
                else:
                    # This is for GoogLeNet
                    param = torch.reshape(param, [param.shape[0], perm_in.shape[0], -1])
                    param = param.permute(0, 2, 1)  # B, -1, C
                    param = torch.matmul(param, perm_in.t())
                    param = param.permute(0, 2, 1)
                    param = torch.flatten(param, start_dim=1)
            if perm_out is not None:
                param = torch.matmul(perm_out, param)
        else:
            param = param.permute(2, 3, 0, 1)
            if perm_in is not None:
                param = torch.matmul(param, perm_in.t())
            if perm_out is not None:
                param = torch.matmul(perm_out, param)
            param = param.permute(2, 3, 0, 1)
        return param


class Linear(CurveModule):
    def __init__(self, in_features, out_features, fix_points, bias=True):
        super(Linear, self).__init__(fix_points, ('weight', 'bias'))
        self.in_features = in_features
        self.out_features = out_features

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'weight_%d' % i,
                Parameter(torch.Tensor(out_features, in_features), requires_grad=not fixed)
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(out_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        for i in range(self.num_bends):
            getattr(self, 'weight_%d' % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t, orient=None):
        weight_t, bias_t = self.compute_weights_t(coeffs_t, orient=orient)
        return F.linear(input, weight_t, bias_t)


class Conv2d(CurveModule):
    def __init__(self, in_channels, out_channels, kernel_size, fix_points, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(fix_points, ('weight', 'bias'))
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'weight_%d' % i,
                Parameter(
                    torch.Tensor(out_channels, in_channels // groups, *kernel_size),
                    requires_grad=not fixed
                )
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(out_channels), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        for i in range(self.num_bends):
            getattr(self, 'weight_%d' % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t, orient=None):
        weight_t, bias_t = self.compute_weights_t(coeffs_t, orient=orient)
        return F.conv2d(input, weight_t, bias_t, self.stride,
                        self.padding, self.dilation, self.groups)


class _BatchNorm(CurveModule):
    _version = 2

    def __init__(self, num_features, fix_points, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__(fix_points, ('weight', 'bias'))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    'weight_%d' % i,
                    Parameter(torch.Tensor(num_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('weight_%d' % i, None)
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(num_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            for i in range(self.num_bends):
                getattr(self, 'weight_%d' % i).data.uniform_()
                getattr(self, 'bias_%d' % i).data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input, coeffs_t, orient=None):
        self._check_input_dim(input)

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        else:
            exponential_average_factor = 0.0
        weight_t, bias_t = self.compute_weights_t(coeffs_t, orient=orient)
        return F.batch_norm(
            input, self.running_mean, self.running_var, weight_t, bias_t,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class CurveNet(Module):
    def __init__(self, num_classes, device, curve, architecture, num_bends, fix_start=True, fix_end=True,
                 architecture_kwargs={}, act_ref=None):
        super(CurveNet, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.num_bends = num_bends
        self.fix_points = [fix_start] + [False] * (self.num_bends - 2) + [fix_end]

        self.curve = curve
        self.architecture = architecture

        self.l2 = 0.0
        self.coeff_layer = self.curve(self.num_bends)
        self.net = self.architecture(num_classes, device, fix_points=self.fix_points, curve_class="curves_pam",
                                     **architecture_kwargs)
        self.curve_learnable_params = nn.ParameterList()
        for param in self.net.parameters():
            if param.requires_grad:
                self.curve_learnable_params.append(param)
        self.permutations = self.create_permutation_matrices(act_ref=act_ref)
        self.curve_modules = []
        for module in self.net.modules():
            if issubclass(module.__class__, CurveModule):
                self.curve_modules.append(module)

    def create_permutation_matrices(self, act_ref=None):
        time_init = time.time() 
        permutations = nn.ParameterList()
        if type(self.net).__name__.startswith('ResNet'):
            id_mat = nn.Parameter(torch.eye(self.net.layers[0][0].in_channels), requires_grad=False)
            permutations.append(id_mat)
            id_mat = nn.Parameter(torch.eye(self.net.layers[0][0].out_channels), requires_grad=False)
            permutations.append(id_mat)
            for block in self.net.layers[1:]:
                for res_unit in block:
                    id_mat = nn.Parameter(torch.eye(res_unit.conv1.out_channels), requires_grad=True)
                    permutations.append(id_mat)

                    id_mat = nn.Parameter(torch.eye(res_unit.conv2.out_channels), requires_grad=True)
                    permutations.append(id_mat)
            id_mat = nn.Parameter(torch.eye(self.num_classes), requires_grad=False)
            permutations.append(id_mat)
        elif type(self.net).__name__.startswith('GoogLeNet'):
            # need matching here for reference
            permutations = nn.ModuleList()
            id_mat = nn.Parameter(torch.eye(3), requires_grad=False)
            permutations.append(nn.ParameterList([id_mat]))
            for act in act_ref:
                temp = nn.ParameterList()
                if isinstance(act, list):
                    for sub_act in act:
                        temp.append(nn.Parameter(torch.eye(sub_act.shape[0]), requires_grad=True))
                    permutations.append(temp)
                else:
                    temp.append(nn.Parameter(torch.eye(act.shape[0]), requires_grad=True))
                    permutations.append(temp)
            temp = nn.ParameterList()
            temp.append(nn.Parameter(torch.eye(self.num_classes), requires_grad=False))
            permutations.append(temp)
        else:
            id_mat = nn.Parameter(torch.eye(self.net.layers[0].conv.in_channels), requires_grad=False)
            permutations.append(id_mat)
            for i, layer in enumerate(self.net.layers):
                id_mat = nn.Parameter(torch.eye(layer.conv.out_channels), requires_grad=True)
                permutations.append(id_mat)

            id_mat = nn.Parameter(torch.eye(self.num_classes), requires_grad=False)
            permutations.append(id_mat)
        print('Permutations created, Time elapsed %.2fs' % (time.time() - time_init)) 
        return permutations

    def import_base_parameters(self, base_model, index):
        parameters = list(self.net.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            parameter.data.copy_(base_parameter.data)

    def import_base_buffers(self, base_model):
        for buffer, base_buffer in zip(self.net._all_buffers(), base_model._all_buffers()):
            buffer.data.copy_(base_buffer.data)

    def export_base_parameters(self, base_model, index):

        parameters = list(self.net.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            base_parameter.data.copy_(parameter.data)

    def init_linear(self):
        parameters = list(self.net.parameters())
        for i in range(0, len(parameters), self.num_bends):
            weights = parameters[i:i + self.num_bends]
            for j in range(1, self.num_bends - 1):
                alpha = j * 1.0 / (self.num_bends - 1)
                weights[j].data.copy_(alpha * weights[-1].data + (1.0 - alpha) * weights[0].data)

    def init_zero(self):
        parameters = list(self.net.parameters())
        for i in range(0, len(parameters), self.num_bends):
            weights = parameters[i:i + self.num_bends]
            for j in range(1, self.num_bends - 1):
                weights[j].data.copy_(0.0 * weights[0].data)

    def weights(self, t):
        coeffs_t = self.coeff_layer(t)
        weights = []
        for module in self.curve_modules:
            weights.extend([w for w in module.compute_weights_t(coeffs_t) if w is not None])
        return np.concatenate([w.detach().cpu().numpy().ravel() for w in weights])

    def compute_new_model(self, base_model, loader, t):
        models_base = [None] * self.num_bends
        states_base = [None] * self.num_bends
        for i in range(self.num_bends):
            models_base[i] = base_model(self.num_classes, self.device)
            self.export_base_parameters(models_base[i], i)
            models_base[i].to(self.device)
            states_base[i] = models_base[i].state_dict()

        coeffs_t = self.coeff_layer(t)

        state_t = copy.deepcopy(states_base[0])
        for i in state_t:
            state_t[i] = sum([coeffs_t[j] * states_base[j][i] for j in range(self.num_bends)])
        model_new = base_model(self.num_classes, self.device)
        model_new.load_state_dict(state_t)
        model_new.to(self.device)
        utils.update_bn(loader, model_new)
        return model_new

    def _compute_l2(self):
        self.l2 = sum(module.l2 for module in self.curve_modules)

    def weight_permutation(self, model_2=None):
        orient = [None] * 3
        if type(self.net).__name__.startswith('ResNet'):
            idx = 0
            orient[0] = torch.nonzero(self.permutations[idx], as_tuple=True)[1]
            orient[1] = torch.nonzero(self.permutations[idx + 1], as_tuple=True)[1]
            for name, param in self.net.layers[0].named_parameters():
                if name.endswith(str(self.num_bends - 1)):
                    param.data = param.data[orient[1]]
                    if param.ndim > 1:
                        param.data = param.data[:, orient[0]]
            for block in self.net.layers[1:]:
                for res_unit in block:
                    idx += 1
                    orient[0] = torch.nonzero(self.permutations[2 * idx - 1], as_tuple=True)[1]
                    orient[1] = torch.nonzero(self.permutations[2 * idx], as_tuple=True)[1]
                    orient[2] = torch.nonzero(self.permutations[2 * idx + 1], as_tuple=True)[1]
                    for name, param in res_unit.named_parameters():
                        if name.endswith(str(self.num_bends - 1)):
                            if name.startswith('conv1') or name.startswith('bn1'):
                                param.data = param.data[orient[1]]
                                if param.ndim > 1:
                                    param.data = param.data[:, orient[0]]
                            if name.startswith('conv2') or name.startswith('bn2'):
                                param.data = param.data[orient[2]]
                                if param.ndim > 1:
                                    param.data = param.data[:, orient[1]]
                            if name.startswith('downsample'):
                                param.data = param.data[orient[2]]
                                if param.ndim > 1:
                                    param.data = param.data[:, orient[0]]
            orient[0] = torch.nonzero(self.permutations[-2], as_tuple=True)[1]
            orient[1] = torch.nonzero(self.permutations[-1], as_tuple=True)[1]
            for name, param in self.net.fc.named_parameters():
                if name.endswith(str(self.num_bends - 1)) and param.ndim > 1:
                    param.data = param.data[orient[1]]
                    param.data = param.data[:, orient[0]]
        elif type(self.net).__name__.startswith('GoogLeNet'):
            match = [] 
            for perm_list in self.permutations[1:-1]:
                if len(perm_list) == 1:
                    orient = torch.nonzero(perm_list[0], as_tuple=True)[1] 
                    match.append(orient.cpu())
                else:
                    sub_match = []
                    for sub_perm in perm_list:
                        orient = torch.nonzero(sub_perm, as_tuple=True)[1]            
                        sub_match.append(orient.cpu())
                    match.append(sub_match) 
            self.export_base_parameters(model_2, 2)
            model_2.align_inception(match)
            self.import_base_parameters(model_2, 2)
        else:
            for idx, layer in enumerate(self.net.layers):
                orient[0] = torch.nonzero(self.permutations[idx], as_tuple=True)[1]
                orient[1] = torch.nonzero(self.permutations[idx + 1], as_tuple=True)[1]
                for name, param in layer.named_parameters():
                    if name.endswith(str(self.num_bends - 1)):
                        param.data = param.data[orient[1]]
                        if param.ndim > 1:
                            param.data = param.data[:, orient[0]]
            orient[0] = torch.nonzero(self.permutations[-2], as_tuple=True)[1]
            orient[1] = torch.nonzero(self.permutations[-1], as_tuple=True)[1]
            for name, param in self.net.dense.named_parameters():
                if name.endswith(str(self.num_bends - 1)) and param.ndim > 1:
                    param.data = param.data[orient[1]]
                    param.data = param.data[:, orient[0]]

    def forward(self, input, t=None, perm_train=False):
        if t is None:
            t = input.data.new(1).uniform_(0.0, 1.0)
        coeffs_t = self.coeff_layer(t)
        if not perm_train:
            output = self.net(input, coeffs_t)
        else:
            output = self.net(input, coeffs_t, perm=self.permutations)
        self._compute_l2()
        return output


def l2_regularizer(weight_decay):
    return lambda model: 0.5 * weight_decay * model.l2
