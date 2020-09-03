import warnings
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from torch.jit.annotations import Optional, Tuple
from torch import Tensor
from utils import alignment
from models import curves, curves_pam


model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


class GoogLeNetBase(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self, num_classes=100, aux_logits=True, transform_input=False, init_weights=True,
                 blocks=None, device=None):
        super(GoogLeNetBase, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.device = device
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=1, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = conv_block(64, 64, kernel_size=1)

        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        # type: (Tensor) -> Tensor
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def align_inception(self, alignments):
        align(self.conv1, tail=None, head=alignments[0])
        align(self.conv2, tail=alignments[0], head=alignments[1])
        align(self.conv3, tail=alignments[1], head=alignments[2])
        next_tail3a = self.align_inception_block(self.inception3a, tail=alignments[2], heads=alignments[3])
        next_tail3b = self.align_inception_block(self.inception3b, tail=next_tail3a, heads=alignments[4])

        next_tail4a = self.align_inception_block(self.inception4a, tail=next_tail3b, heads=alignments[5])

        self.align_aux_block(self.aux1, tail=next_tail4a, heads=alignments[6])

        next_tail4b = self.align_inception_block(self.inception4b, tail=next_tail4a, heads=alignments[7])
        next_tail4c = self.align_inception_block(self.inception4c, tail=next_tail4b, heads=alignments[8])
        next_tail4d = self.align_inception_block(self.inception4d, tail=next_tail4c, heads=alignments[9])

        self.align_aux_block(self.aux2, tail=next_tail4d, heads=alignments[10])

        next_tail4e = self.align_inception_block(self.inception4e, tail=next_tail4d, heads=alignments[11])

        next_tail5a = self.align_inception_block(self.inception5a, tail=next_tail4e, heads=alignments[12])
        next_tail5b = self.align_inception_block(self.inception5b, tail=next_tail5a, heads=alignments[13])

        align(self.fc, tail=next_tail5b, head=None)

    def align_inception_block(self, block, tail, heads):
        # The end of each block is what is listed first, followed by the intermediates
        align(block.branch1, tail, heads[0])
        align(block.branch2[0], tail, heads[1])
        align(block.branch2[1], heads[1], heads[2])
        align(block.branch3[0], tail, heads[3])
        align(block.branch3[1], heads[3], heads[4])
        align(block.branch4, tail, heads[5])

        heads2_a = [i + len(heads[0]) for i in heads[2]]
        heads4_a = [i + len(heads[0]) + len(heads[2]) for i in heads[4]]
        heads5_a = [i + len(heads[0]) + len(heads[2]) + len(heads[4]) for i in heads[5]]
        next_tail = np.concatenate([heads[0], heads2_a, heads4_a, heads5_a]) 
        return next_tail

    def align_aux_block(self, block, tail, heads):
        align(block.conv, tail, heads[0])
        fc1_temp = block.fc1.weight.reshape([block.fc1.weight.shape[0], len(heads[0]),
                                             int((block.fc1.weight.shape[1] / len(heads[0])) ** 0.5),
                                             int((block.fc1.weight.shape[1] / len(heads[0])) ** 0.5)])
        fc1_temp = fc1_temp[heads[1], :]
        fc1_temp = fc1_temp[:, heads[0]] 
        fc1_temp = torch.flatten(fc1_temp, 1) 
        block.fc1.weight.data = fc1_temp
        block.fc1.bias.data = block.fc1.bias[heads[1]]
        align(block.fc2, heads[1], head=None)

    def _forward(self, x, store_int=False):
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        int_results = []

        # N x 3 x 224 x 224
        x = self.conv1(x)

        # N x 64 x 112 x 112  -> N X 64 X 32 X 32
        # x = self.maxpool1(x)

        # This is accumulating the intermediate activations for us to compute statistics with
        if store_int:
            int_act = torch.transpose(x, 0, 1)
            int_act = int_act.reshape(int_act.shape[0], -1)
            int_results.append(int_act)

        # N x 64 x 56 x 56 -> N X 64 X 32 X 32
        x = self.conv2(x)

        if store_int:
            int_act = torch.transpose(x, 0, 1)
            int_act = int_act.reshape(int_act.shape[0], -1)
            int_results.append(int_act)

        # N x 64 x 56 x 56 -> N X 64 X 32 X 32
        x = self.conv3(x)
        # N x 192 x 56 x 56 -> N X 64 X 32 X 32
        x = self.maxpool2(x)

        if store_int:
            int_act = torch.transpose(x, 0, 1)
            int_act = int_act.reshape(int_act.shape[0], -1)
            int_results.append(int_act)

        # N x 192 x 28 x 28 -> N X 64 X 16 X 16
        x, incept3a_acts = self.inception3a(x, store_int=store_int)
        if store_int:
            int_results.append(incept3a_acts)

        # N x 256 x 28 x 28 -> N X 64 X 16 X 16
        x, incept3b_acts = self.inception3b(x, store_int=store_int)
        if store_int:
            int_results.append(incept3b_acts)

        # N x 480 x 28 x 28 -> N X 64 X 16 X 16
        x = self.maxpool3(x)

        # N x 480 x 14 x 14 -> N X 64 X 8 X 8
        x, incept4a_acts = self.inception4a(x, store_int=store_int)
        if store_int:
            int_results.append(incept4a_acts)

        # N x 512 x 14 x 14 -> N X 64 X 8 X 8
        aux1 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux1 is not None:
            if self.training or store_int:
                aux1, aux1_acts = self.aux1(x, store_int=store_int)
                if store_int:
                    int_results.append(aux1_acts)

        x, incept4b_acts = self.inception4b(x, store_int=store_int)
        if store_int:
            int_results.append(incept4b_acts)

        # N x 512 x 14 x 14 -> N X 64 X 8 X 8
        x, incept4c_acts = self.inception4c(x, store_int=store_int)
        if store_int:
            int_results.append(incept4c_acts)

        # N x 512 x 14 x 14 -> N X 64 X 8 X 8
        x, incept4d_acts = self.inception4d(x, store_int=store_int)
        if store_int:
            int_results.append(incept4d_acts)
        # N x 528 x 14 x 14 -> N X 64 X 8 X 8
        aux2 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux2 is not None:
            if self.training or store_int:
                aux2, aux2_acts = self.aux2(x, store_int=store_int)
                if store_int:
                    int_results.append(aux2_acts)

        x, incept4e_acts = self.inception4e(x, store_int=store_int)
        if store_int:
            int_results.append(incept4e_acts)
        # N x 832 x 14 x 14 -> N X 64 X 8 X 8
        # x = self.maxpool4(x)
        # N x 832 x 7 x 7 -> N X 64 X 8 X 8
        x, incept5a_acts = self.inception5a(x, store_int=store_int)
        if store_int:
            int_results.append(incept5a_acts)

        # N x 832 x 7 x 7 -> N X 64 X 8 X 8
        x, incept5b_acts = self.inception5b(x, store_int=store_int)
        if store_int:
            int_results.append(incept5b_acts)
        # N x 1024 x 7 x 7 -> N X 64 X 8 X 8

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux2, aux1, int_results

    @torch.jit.unused
    def eager_outputs(self, x, aux2, aux1):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> GoogLeNetOutputs
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def forward(self, x, store_int=False):
        # type: (Tensor) -> GoogLeNetOutputs
        x = self._transform_input(x)
        x, aux1, aux2, int_results = self._forward(x, store_int=store_int)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            if store_int:
                return GoogLeNetOutputs(x, aux2, aux1), int_results
            else:
                return GoogLeNetOutputs(x, aux2, aux1)
        else:
            if store_int:
                return self.eager_outputs(x, aux2, aux1), int_results
            else:
                return self.eager_outputs(x, aux2, aux1)


class GoogLeNetCurve(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self, num_classes, device, fix_points, aux_logits=True, transform_input=False, init_weights=True,
                 blocks=None, curve_class="curves"):
        super(GoogLeNetCurve, self).__init__()
        if blocks is None:
            blocks = [BasicConv2dCurve, InceptionCurve, InceptionAuxCurve]
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.fix_points = fix_points
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.device = device

        curve_class = importlib.import_module("models." + curve_class)

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=1, padding=3, fix_points=fix_points, curve_class=curve_class)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = conv_block(64, 64, kernel_size=1, fix_points=fix_points, curve_class=curve_class)

        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1, fix_points=fix_points, curve_class=curve_class)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32, fix_points=fix_points, curve_class=curve_class)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64, fix_points=fix_points, curve_class=curve_class)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64, fix_points=fix_points, curve_class=curve_class)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64, fix_points=fix_points, curve_class=curve_class)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64, fix_points=fix_points, curve_class=curve_class)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64, fix_points=fix_points, curve_class=curve_class)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128, fix_points=fix_points, curve_class=curve_class)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128, fix_points=fix_points, curve_class=curve_class)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128, fix_points=fix_points, curve_class=curve_class)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes, fix_points=fix_points, curve_class=curve_class)
            self.aux2 = inception_aux_block(528, num_classes, fix_points=fix_points, curve_class=curve_class)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = curve_class.Linear(1024, num_classes, fix_points=fix_points)

        if init_weights:
            self._initialize_weights(curve_class)

    def _initialize_weights(self, curve_class="curves"):
        for m in self.modules():
            if isinstance(m, curve_class.Conv2d) or isinstance(m, curve_class.Linear):

                for i in range(len(self.fix_points)):
                    nn.init.xavier_normal_(getattr(m, 'weight_%d' % i))
            elif isinstance(m, curve_class.BatchNorm2d):
                for i in range(len(self.fix_points)):
                    nn.init.constant_(getattr(m, 'weight_%d' % i), 1)
                    nn.init.constant_(getattr(m, 'bias_%d' % i), 0)

    def _transform_input(self, x):
        # type: (Tensor) -> Tensor
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x, coeffs_t, perm=None):
        # type: (Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
        # N x 3 x 224 x 224

        if perm is None:
            x = self.conv1(x, coeffs_t)

            # N x 64 x 112 x 112  -> N X 64 X 32 X 32
            # x = self.maxpool1(x)

            # This is accumulating the intermediate activations for us to compute statistics with
            # N x 64 x 56 x 56 -> N X 64 X 32 X 32
            x = self.conv2(x, coeffs_t)

            # N x 64 x 56 x 56 -> N X 64 X 32 X 32
            x = self.conv3(x, coeffs_t)
            # N x 192 x 56 x 56 -> N X 64 X 32 X 32
            x = self.maxpool2(x)

            # N x 192 x 28 x 28 -> N X 64 X 16 X 16
            x = self.inception3a(x, coeffs_t)

            # N x 256 x 28 x 28 -> N X 64 X 16 X 16
            x = self.inception3b(x, coeffs_t)

            # N x 480 x 28 x 28 -> N X 64 X 16 X 16
            x = self.maxpool3(x)

            # N x 480 x 14 x 14 -> N X 64 X 8 X 8
            x = self.inception4a(x, coeffs_t)

            # N x 512 x 14 x 14 -> N X 64 X 8 X 8
            aux1 = torch.jit.annotate(Optional[Tensor], None)
            if self.aux1 is not None:
                if self.training:
                    aux1 = self.aux1(x, coeffs_t)

            x = self.inception4b(x, coeffs_t)

            # N x 512 x 14 x 14 -> N X 64 X 8 X 8
            x = self.inception4c(x, coeffs_t)

            # N x 512 x 14 x 14 -> N X 64 X 8 X 8
            x = self.inception4d(x, coeffs_t)
            # N x 528 x 14 x 14 -> N X 64 X 8 X 8
            aux2 = torch.jit.annotate(Optional[Tensor], None)
            if self.aux2 is not None:
                if self.training:
                    aux2 = self.aux2(x, coeffs_t)

            x = self.inception4e(x, coeffs_t)
            # N x 832 x 14 x 14 -> N X 64 X 8 X 8
            # x = self.maxpool4(x)
            # N x 832 x 7 x 7 -> N X 64 X 8 X 8
            x = self.inception5a(x, coeffs_t)

            # N x 832 x 7 x 7 -> N X 64 X 8 X 8
            x = self.inception5b(x, coeffs_t)
            # N x 1024 x 7 x 7 -> N X 64 X 8 X 8

            x = self.avgpool(x)
            # N x 1024 x 1 x 1
            x = torch.flatten(x, 1)
            # N x 1024
            x = self.dropout(x)
            x = self.fc(x, coeffs_t)
            # N x 1000 (num_classes)
        else:
            x = self.conv1(x, coeffs_t, orient=[perm[0][0], perm[1][0]])

            # N x 64 x 112 x 112  -> N X 64 X 32 X 32
            # x = self.maxpool1(x)

            # This is accumulating the intermediate activations for us to compute statistics with
            # N x 64 x 56 x 56 -> N X 64 X 32 X 32
            x = self.conv2(x, coeffs_t, orient=[perm[1][0], perm[2][0]])

            # N x 64 x 56 x 56 -> N X 64 X 32 X 32
            x = self.conv3(x, coeffs_t, orient=[perm[2][0], perm[3][0]])
            # N x 192 x 56 x 56 -> N X 64 X 32 X 32
            x = self.maxpool2(x)

            # N x 192 x 28 x 28 -> N X 64 X 16 X 16
            x = self.inception3a(x, coeffs_t, orient=[perm[3][0], perm[4]])

            perm4_in = self.permute_inception_block(perm[4])

            # N x 256 x 28 x 28 -> N X 64 X 16 X 16
            x = self.inception3b(x, coeffs_t, orient=[perm4_in, perm[5]])
            perm5_in = self.permute_inception_block(perm[5])

            # N x 480 x 28 x 28 -> N X 64 X 16 X 16
            x = self.maxpool3(x)

            # N x 480 x 14 x 14 -> N X 64 X 8 X 8
            x = self.inception4a(x, coeffs_t, orient=[perm5_in, perm[6]])
            perm6_in = self.permute_inception_block(perm[6])

            # N x 512 x 14 x 14 -> N X 64 X 8 X 8
            aux1 = torch.jit.annotate(Optional[Tensor], None)
            if self.aux1 is not None:
                if self.training:
                    aux1 = self.aux1(x, coeffs_t, orient=[perm6_in, perm[7]])

            x = self.inception4b(x, coeffs_t, orient=[perm6_in, perm[8]])
            perm8_in = self.permute_inception_block(perm[8])

            # N x 512 x 14 x 14 -> N X 64 X 8 X 8
            x = self.inception4c(x, coeffs_t, orient=[perm8_in, perm[9]])
            perm9_in = self.permute_inception_block(perm[9])

            # N x 512 x 14 x 14 -> N X 64 X 8 X 8
            x = self.inception4d(x, coeffs_t, orient=[perm9_in, perm[10]])
            perm10_in = self.permute_inception_block(perm[10])

            # N x 528 x 14 x 14 -> N X 64 X 8 X 8
            aux2 = torch.jit.annotate(Optional[Tensor], None)
            if self.aux2 is not None:
                if self.training:
                    aux2 = self.aux2(x, coeffs_t, orient=[perm10_in, perm[11]])

            x = self.inception4e(x, coeffs_t, orient=[perm10_in, perm[12]])
            perm12_in = self.permute_inception_block(perm[12])

            # N x 832 x 14 x 14 -> N X 64 X 8 X 8
            # x = self.maxpool4(x)
            # N x 832 x 7 x 7 -> N X 64 X 8 X 8
            x = self.inception5a(x, coeffs_t, orient=[perm12_in, perm[13]])
            perm13_in = self.permute_inception_block(perm[13])

            # N x 832 x 7 x 7 -> N X 64 X 8 X 8
            x = self.inception5b(x, coeffs_t, orient=[perm13_in, perm[14]])
            perm14_in = self.permute_inception_block(perm[14])
            # N x 1024 x 7 x 7 -> N X 64 X 8 X 8

            x = self.avgpool(x)
            # N x 1024 x 1 x 1
            x = torch.flatten(x, 1)
            # N x 1024
            x = self.dropout(x)
            x = self.fc(x, coeffs_t, orient=[perm14_in, perm[15][0]])
            # N x 1000 (num_classes)
        return x, aux2, aux1

    def permute_inception_block(self, perm):
        perm_in_shape = [perm[0].shape[0] + perm[2].shape[0] \
                          + perm[4].shape[0] + perm[5].shape[0],
                          perm[0].shape[1] + perm[2].shape[1] \
                          + perm[4].shape[1] + perm[5].shape[1]]
        perm_in = torch.zeros(perm_in_shape, device=self.device)
        idx0 = 0
        idx1 = 0
        for i in [0, 2, 4, 5]:
            perm_in[idx0: idx0 + perm[i].shape[0], idx1: idx1 + perm[i].shape[1]] \
                = perm[i]
            idx0 += perm[i].shape[0]
            idx1 += perm[i].shape[0]
        return perm_in

    @torch.jit.unused
    def eager_outputs(self, x, aux2, aux1):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> GoogLeNetOutputs
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def forward(self, x, coeffs_t, perm=None):
        # type: (Tensor) -> GoogLeNetOutputs
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x, coeffs_t, perm=perm)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)

    def import_base_parameters(self, base_model, index):
        parameters = list(self.parameters())[index::len(self.fix_points)]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            parameter.data.copy_(base_parameter.data)


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.ModuleList()
        self.branch2.append(conv_block(in_channels, ch3x3red, kernel_size=1))
        self.branch2.append(conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1))

        self.branch3 = nn.ModuleList()
        self.branch3.append(conv_block(in_channels, ch5x5red, kernel_size=1))
        self.branch3.append(conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1))

        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
        self.branch4 = conv_block(in_channels, pool_proj, kernel_size=1)

    def _forward(self, x, store_int=False):
        int_results = []

        branch1 = self.branch1(x)
        if store_int:
            int_act = torch.transpose(branch1, 0, 1)
            int_act = int_act.reshape(int_act.shape[0], -1)
            int_results.append(int_act)

        branch2 = self.branch2[0](x)
        if store_int:
            int_act = torch.transpose(branch2, 0, 1)
            int_act = int_act.reshape(int_act.shape[0], -1)
            int_results.append(int_act)
        branch2 = self.branch2[1](branch2)
        if store_int:
            int_act = torch.transpose(branch2, 0, 1)
            int_act = int_act.reshape(int_act.shape[0], -1)
            int_results.append(int_act)

        branch3 = self.branch3[0](x)
        if store_int:
            int_act = torch.transpose(branch3, 0, 1)
            int_act = int_act.reshape(int_act.shape[0], -1)
            int_results.append(int_act)
        branch3 = self.branch3[1](branch3)
        if store_int:
            int_act = torch.transpose(branch3, 0, 1)
            int_act = int_act.reshape(int_act.shape[0], -1)
            int_results.append(int_act)

        branch4 = self.branch4_pool(x)
        branch4 = self.branch4(branch4)
        if store_int:
            int_act = torch.transpose(branch4, 0, 1)
            int_act = int_act.reshape(int_act.shape[0], -1)
            int_results.append(int_act)

        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs, int_results

    def forward(self, x, store_int):
        outputs, int_results = self._forward(x, store_int=store_int)
        return outputs, int_results


class InceptionCurve(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 fix_points, conv_block=None, curve_class="curves"):
        super(InceptionCurve, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2dCurve
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1, fix_points=fix_points, curve_class=curve_class)

        self.branch2 = nn.ModuleList()
        self.branch2.append(conv_block(in_channels, ch3x3red, kernel_size=1, fix_points=fix_points, curve_class=curve_class))
        self.branch2.append(conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1, fix_points=fix_points, curve_class=curve_class))

        self.branch3 = nn.ModuleList()
        self.branch3.append(conv_block(in_channels, ch5x5red, kernel_size=1, fix_points=fix_points, curve_class=curve_class))
        self.branch3.append(conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1, fix_points=fix_points,
                                       curve_class=curve_class))  # There was bug, originally a 3
        # Factorize this convolution to save time and parameters

        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
        self.branch4 = conv_block(in_channels, pool_proj, kernel_size=1, fix_points=fix_points, curve_class=curve_class)

    def _forward(self, x, coeffs_t, orient=None):
        if orient is None:
            branch1 = self.branch1(x, coeffs_t)
            branch2 = self.branch2[0](x, coeffs_t)
            branch2 = self.branch2[1](branch2, coeffs_t)
            branch3 = self.branch3[0](x, coeffs_t)
            branch3 = self.branch3[1](branch3, coeffs_t)
            branch4 = self.branch4_pool(x)
            branch4 = self.branch4(branch4, coeffs_t)
            outputs = [branch1, branch2, branch3, branch4]
        else:
            branch1 = self.branch1(x, coeffs_t, orient=[orient[0], orient[1][0]])
            branch2 = self.branch2[0](x, coeffs_t, orient=[orient[0], orient[1][1]])
            branch2 = self.branch2[1](branch2, coeffs_t, orient=[orient[1][1], orient[1][2]])
            branch3 = self.branch3[0](x, coeffs_t, orient=[orient[0], orient[1][3]])
            branch3 = self.branch3[1](branch3, coeffs_t, orient=[orient[1][3], orient[1][4]])
            branch4 = self.branch4_pool(x)
            branch4 = self.branch4(branch4, coeffs_t, orient=[orient[0], orient[1][5]])
            outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x, coeffs_t, orient=None):
        outputs = self._forward(x, coeffs_t, orient=orient)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x, store_int=False):
        int_results = []
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        if store_int:
            int_act = torch.transpose(x, 0, 1)
            int_act = int_act.reshape(int_act.shape[0], -1)
            int_results.append(int_act)

        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        if store_int:
            int_act = torch.transpose(x, 0, 1)
            int_act = int_act.reshape(int_act.shape[0], -1)
            int_results.append(int_act)

        # N x 1024
        x = F.dropout(x, 0.7, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x, int_results


class InceptionAuxCurve(nn.Module):

    def __init__(self, in_channels, num_classes, fix_points, conv_block=None, curve_class="curves"):
        super(InceptionAuxCurve, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2dCurve
        self.conv = conv_block(in_channels, 128, fix_points=fix_points, kernel_size=1, curve_class=curve_class)

        self.fc1 = curve_class.Linear(2048, 1024, fix_points=fix_points)
        self.fc2 = curve_class.Linear(1024, num_classes, fix_points=fix_points)

    def forward(self, x, coeffs_t, orient=None):
        if orient is None:
            # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
            x = F.adaptive_avg_pool2d(x, (4, 4))
            # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
            x = self.conv(x, coeffs_t)
            # N x 128 x 4 x 4
            x = torch.flatten(x, 1)
            # N x 2048
            x = F.relu(self.fc1(x, coeffs_t), inplace=True)

            # N x 1024
            x = F.dropout(x, 0.7, training=self.training)
            # N x 1024
            x = self.fc2(x, coeffs_t)
            # N x 1000 (num_classes)
        else:
            # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
            x = F.adaptive_avg_pool2d(x, (4, 4))
            # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
            x = self.conv(x, coeffs_t, orient=[orient[0], orient[1][0]])
            # N x 128 x 4 x 4

            x = torch.flatten(x, 1)
            # N x 2048
            x = F.relu(self.fc1(x, coeffs_t, orient=[orient[1][0], orient[1][1]]), inplace=True)

            # N x 1024
            x = F.dropout(x, 0.7, training=self.training)
            # N x 1024
            x = self.fc2(x, coeffs_t, orient=[orient[1][1], None])
            # N x 1000 (num_classes)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class BasicConv2dCurve(nn.Module):
    def __init__(self, in_channels, out_channels, fix_points, curve_class="curves", **kwargs):
        super(BasicConv2dCurve, self).__init__()
        self.conv = curve_class.Conv2d(in_channels, out_channels, fix_points=fix_points, bias=False, **kwargs)
        self.bn = curve_class.BatchNorm2d(out_channels, eps=0.001, fix_points=fix_points)

    def forward(self, x, coeffs_t, **kwargs):
        x = self.conv(x, coeffs_t, **kwargs)
        x = self.bn(x, coeffs_t, **kwargs)
        return F.relu(x, inplace=True)


def align(layer, tail=None, head=None):
    for name, m in layer.named_modules():
        if tail is not None:
            alignment.align_weights_tail(m, tail)
        if head is not None:
            alignment.align_weights_head(m, head)


class GoogLeNet:
    base = GoogLeNetBase
    curve = GoogLeNetCurve
    kwargs = {}
