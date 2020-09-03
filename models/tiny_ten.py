import torch.nn as nn
import torch.nn.functional as F
from models import curves, curves_pam
import importlib

__all__ = ['TinyTen']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, bias=True, stride=1, padding=0):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel, bias=bias, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, pre_act=False):
        out = self.conv(x)
        out = self.bn(out)
        if not pre_act:
            out = self.act(out)

        return out


class BasicBlockCurve(nn.Module):
    def __init__(self, fix_points, in_planes, out_planes, kernel, bias=True, stride=1, padding=0, curve_class=curves):
        super(BasicBlockCurve, self).__init__()
        self.conv = curve_class.Conv2d(in_planes, out_planes, kernel_size=kernel, fix_points=fix_points, stride=stride,
                                       padding=padding, bias=bias)
        self.bn = curve_class.BatchNorm2d(out_planes, fix_points=fix_points)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, coeffs_t, **kwargs):
        out = self.conv(x, coeffs_t, **kwargs)
        out = self.bn(out, coeffs_t, **kwargs)
        out = self.act(out)

        return out


class TinyTenBase(nn.Module):
    def __init__(self, num_classes, device, bias=True, in_channels=3):
        super(TinyTenBase, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.layers.append(BasicBlock(in_channels, 16, 3, bias=bias, padding=1))
        self.layers.append(BasicBlock(16, 16, 3, bias=bias, padding=1))
        self.layers.append(BasicBlock(16, 32, 3, bias=bias, stride=2, padding=1))
        self.layers.append(BasicBlock(32, 32, 3, bias=bias, padding=1))
        self.layers.append(BasicBlock(32, 32, 3, bias=bias, padding=1))
        self.layers.append(BasicBlock(32, 64, 3, bias=bias, stride=2, padding=1))
        if num_classes == 10 or num_classes == 100:
            self.layers.append(BasicBlock(64, 64, 3, bias=bias))
            self.layers.append(BasicBlock(64, 64, 1, bias=bias))
            self.dense = nn.Linear(64, num_classes)
        elif num_classes == 200:
            self.layers.append(BasicBlock(64, 64, 3, bias=bias)) 
            self.layers.append(BasicBlock(64, 64, 1, bias=bias))
            self.dense = nn.Linear(64, num_classes)

        # if num_classes == 100:
        #      self.drop = nn.Dropout(p=0.2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if bias or isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

        out = F.avg_pool2d(x, kernel_size=x.shape[-1])
        out = out.view(out.shape[0], -1)
        # if hasattr(self, 'drop'):
        #      out = self.drop(out)

        logits = self.dense(out)
        return logits

    def get_activation(self, x, idx_act, idx_warmstart=None, pre_act=False):
        if idx_warmstart is None:
            idx_warmstart = -1
        for idx, layer in enumerate(self.layers[idx_warmstart+1:idx_act+1]):
            x = layer(x, pre_act=pre_act)
            if idx != idx_act and pre_act:
                x = F.relu(x)

        if idx_act + 1 > len(self.layers):
            x = F.avg_pool2d(x, kernel_size=x.shape[-1])
            x = x.view(x.shape[0], -1)
            x = self.dense(x)
        return x


class TinyTenCurve(nn.Module):
    def __init__(self, num_classes, device, fix_points, bias=True, curve_class="curves"):
        super(TinyTenCurve, self).__init__()
        self.device = device
        self.fix_points = fix_points
        self.layers = nn.ModuleList()

        curve_class = importlib.import_module("models." + curve_class)

        self.layers.append(BasicBlockCurve(fix_points, 3, 16, 3, bias=bias, padding=1, curve_class=curve_class))
        self.layers.append(BasicBlockCurve(fix_points, 16, 16, 3, bias=bias, padding=1, curve_class=curve_class))
        self.layers.append(BasicBlockCurve(fix_points, 16, 32, 3, bias=bias, stride=2, padding=1,
                                           curve_class=curve_class))
        self.layers.append(BasicBlockCurve(fix_points, 32, 32, 3, bias=bias, padding=1, curve_class=curve_class))
        self.layers.append(BasicBlockCurve(fix_points, 32, 32, 3, bias=bias, padding=1, curve_class=curve_class))
        self.layers.append(BasicBlockCurve(fix_points, 32, 64, 3, bias=bias, stride=2, padding=1,
                                           curve_class=curve_class))
        if num_classes == 10 or num_classes == 100:
            self.layers.append(BasicBlockCurve(fix_points, 64, 64, 3, bias=bias, curve_class=curve_class))
            self.layers.append(BasicBlockCurve(fix_points, 64, 64, 1, bias=bias, curve_class=curve_class))
            self.gap = nn.AvgPool2d(kernel_size=6)
            self.dense = curve_class.Linear(64, num_classes, fix_points=fix_points)
        elif num_classes == 200 or num_classes == 120:
            self.layers.append(BasicBlockCurve(fix_points, 64, 64, 3, bias=bias, stride=2, padding=1, curve_class=curve_class))
            self.layers.append(BasicBlockCurve(fix_points, 64, 64, 1, bias=bias, curve_class=curve_class)) 
            self.gap = nn.AvgPool2d(kernel_size=6)
            self.dense = curve_class.Linear(64, num_classes, fix_points=fix_points)
 
        for m in self.modules():
            if isinstance(m, curve_class.Conv2d) or isinstance(m, curve_class.Linear):
                for i in range(len(fix_points)):
                    nn.init.xavier_normal_(getattr(m, 'weight_%d' % i))
                    if bias or isinstance(m, curve_class.Linear):
                        nn.init.constant_(getattr(m, 'bias_%d' % i), 0)
            elif isinstance(m, curve_class.BatchNorm2d):
                for i in range(len(fix_points)):
                    nn.init.constant_(getattr(m, 'weight_%d' % i), 1)
                    nn.init.constant_(getattr(m, 'bias_%d' % i), 0)

    def forward(self, x, coeffs_t, perm=None):
        for idx, layer in enumerate(self.layers):
            if perm is not None:
                orient = [perm[idx], perm[idx + 1]]
                x = layer(x, coeffs_t, orient=orient)
            else:
                x = layer(x, coeffs_t)

        out = F.avg_pool2d(x, kernel_size=x.shape[-1])
        out = out.view(out.shape[0], -1)

        if perm is not None:
            orient = [perm[-2], perm[-1]]
            logits = self.dense(out, coeffs_t, orient=orient)
        else:
            logits = self.dense(out, coeffs_t)
        return logits

    def get_activation(self, x, coeffs_t, idx_act):
        for layer in self.layers[:idx_act+1]:
            x = layer(x, coeffs_t)
        return x

    def import_base_parameters(self, base_model, index):
        parameters = list(self.parameters())[index::len(self.fix_points)]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            parameter.data.copy_(base_parameter.data)


class TinyTen:
    base = TinyTenBase
    curve = TinyTenCurve
    kwargs = {}
