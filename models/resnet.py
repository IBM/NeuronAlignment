import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import curves, curves_pam
import importlib

__all__ = ['ResNet32']


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False)


def conv3x3curve(in_planes, out_planes, fix_points, stride=1, dilation=1, curve_class=curves):
    return curve_class.Conv2d(in_planes, out_planes, kernel_size=3, fix_points=fix_points, stride=stride,
                              padding=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv1x1curve(in_planes, out_planes, fix_points, stride=1, curve_class=curves):
    """1x1 convolution"""
    return curve_class.Conv2d(in_planes, out_planes, kernel_size=1, fix_points=fix_points, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out_int = self.relu(out)

        out = self.conv2(out_int)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, out_int


class BasicBlockCurve(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, fix_points, stride=1, downsample=None, norm_layer=None, curve_class=curves):
        super(BasicBlockCurve, self).__init__()
        if norm_layer is None:
            norm_layer = curve_class.BatchNorm2d
        self.conv1 = conv3x3curve(inplanes, planes, fix_points, stride, curve_class=curve_class)
        self.bn1 = norm_layer(planes, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3curve(planes, planes, fix_points, curve_class=curve_class)
        self.bn2 = norm_layer(planes, fix_points=fix_points)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, coeffs_t, orient_all=None, **kwargs):
        id_x = x

        if orient_all is not None and self.downsample is None:
            orient_all[2].data = orient_all[0]

        if orient_all is not None:
            orient = [orient_all[0], orient_all[1]]
            kwargs = {'orient': orient}
        out = self.conv1(x, coeffs_t, **kwargs)
        out = self.bn1(out, coeffs_t, **kwargs)
        out = self.relu(out)

        if orient_all is not None:
            orient = [orient_all[1], orient_all[2]]
            kwargs = {'orient': orient}
        out = self.conv2(out, coeffs_t, **kwargs)
        out = self.bn2(out, coeffs_t, **kwargs)

        if self.downsample is not None:
            if orient_all is not None:
                orient = [orient_all[0], orient_all[2]]
                kwargs = {'orient': orient}
            for layer in self.downsample:
                id_x = layer(id_x, coeffs_t, **kwargs)
        out += id_x
        out = self.relu(out)

        return out


class ResNetBase(nn.Module):

    def __init__(self, num_classes, device, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, block=BasicBlock, layers=[5, 5, 5]):
        super(ResNetBase, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layers.append(nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                         bias=False),
                                         norm_layer(self.inplanes),
                                         nn.ReLU(inplace=True)))

        self.layers.append(self._make_layer(block, 16, layers[0]))
        self.layers.append(self._make_layer(block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0]))
        self.layers.append(self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1]))

        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # if num_classes == 100:
        #     self.drop = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return layers

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            for block in layer:
                x, _ = block(x)

        x = F.avg_pool2d(x, kernel_size=x.shape[-1])
        x = torch.flatten(x, 1)
        # if hasattr(self, 'drop'):
        #     x = self.drop(x)
        x = self.fc(x)

        return x

    def get_activation(self, x, idx_act, idx_warmstart=None):
        x_int = None
        i = 0
        if idx_warmstart is None:
            idx_warmstart = -1
            x = self.layers[0](x)
            if idx_act == 0:
                return x_int, x

        resunit_nums = np.cumsum([5, 5, 5])
        for j, layer in enumerate(self.layers[1:]):
            if idx_warmstart < resunit_nums[j]:
                for block in layer:
                    if idx_warmstart <= i:
                        if i >= idx_act:
                            return x_int, x
                        x, x_int = block(x)
                    i += 1
            else:
                i = resunit_nums[j]
        if i < idx_act:
            x_int = None

            x = F.avg_pool2d(x, kernel_size=x.shape[-1])
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x_int, x


class ResNetCurve(nn.Module):

    def __init__(self, num_classes, device, fix_points, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, block=BasicBlockCurve, layers=[5, 5, 5], curve_class="curves"):
        super(ResNetCurve, self).__init__()

        self.device = device
        self.fix_points = fix_points
        self.layers = nn.ModuleList()

        curve_class = importlib.import_module("models." + curve_class)
        self.curve_class = curve_class

        if norm_layer is None:
            norm_layer = curve_class.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        layer0 = nn.ModuleList()
        layer0.append(curve_class.Conv2d(3, self.inplanes, kernel_size=3, fix_points=fix_points, padding=1,
                                         bias=False))
        layer0.append(norm_layer(self.inplanes, fix_points=fix_points))
        layer0.append(nn.ReLU(inplace=True))

        self.layers.append(layer0)

        self.layers.append(self._make_layer(block, 16, layers[0], fix_points, curve_class=curve_class))
        self.layers.append(self._make_layer(block, 32, layers[1], fix_points, stride=2,
                                            dilate=replace_stride_with_dilation[0], curve_class=curve_class))
        self.layers.append(self._make_layer(block, 64, layers[2], fix_points, stride=2,
                                            dilate=replace_stride_with_dilation[1], curve_class=curve_class))
        self.avgpool = nn.AvgPool2d(8, count_include_pad=False)
        self.fc = curve_class.Linear(64 * block.expansion, num_classes, fix_points)

        for m in self.modules():
            if isinstance(m, curve_class.Conv2d):
                for i in range(len(fix_points)):
                    nn.init.kaiming_normal_(getattr(m, 'weight_%d' % i), mode='fan_out', nonlinearity='relu')
            elif isinstance(m, curve_class.BatchNorm2d):
                nn.init.constant_(getattr(m, 'weight_%d' % i), 1)
                nn.init.constant_(getattr(m, 'bias_%d' % i), 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, fix_points, stride=1, dilate=False, curve_class=curves):
        norm_layer = self._norm_layer
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.ModuleList()
            downsample.append(conv1x1curve(self.inplanes, planes * block.expansion, fix_points, stride,
                                           curve_class=curve_class))
            downsample.append(norm_layer(planes * block.expansion, fix_points=fix_points))

        layers = nn.ModuleList()
        layers.append(block(self.inplanes, planes, fix_points, stride, downsample, norm_layer, curve_class=curve_class))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, fix_points, norm_layer=norm_layer, curve_class=curve_class))

        return layers

    def forward(self, x, coeffs_t, perm=None):
        idx = 0
        for layer in self.layers[0]:
            if perm is not None:
                if isinstance(layer, self.curve_class.Conv2d) or isinstance(layer, self.curve_class.BatchNorm2d):
                    orient = [perm[idx], perm[idx + 1]]
                    x = layer(x, coeffs_t, orient=orient)
                else:
                    x = layer(x)
            else:
                if isinstance(layer, self.curve_class.Conv2d) or isinstance(layer, self.curve_class.BatchNorm2d):
                    x = layer(x, coeffs_t)
                else:
                    x = layer(x)

        for layer in self.layers[1:]:
            for block in layer:
                idx += 1
                if perm is not None:
                    orient_all = [perm[2 * idx - 1], perm[2 * idx], perm[2 * idx + 1]]
                    x = block(x, coeffs_t, orient_all)
                else:
                    x = block(x, coeffs_t)

        x = F.avg_pool2d(x, kernel_size=x.shape[-1])
        x = torch.flatten(x, 1)
        if perm is not None:
            orient = [perm[-2], perm[-1]]
            x = self.fc(x, coeffs_t, orient=orient)
        else:
            x = self.fc(x, coeffs_t)

        return x

    def import_base_parameters(self, base_model, index):
        parameters = list(self.parameters())[index::len(self.fix_points)]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            parameter.data.copy_(base_parameter.data)


class ResNet32:
    base = ResNetBase
    curve = ResNetCurve
    kwargs = {'layers': [5, 5, 5]}
