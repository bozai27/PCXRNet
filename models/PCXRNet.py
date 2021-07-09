import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter

class _BatchAttNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False):
        super(_BatchAttNorm, self).__init__(num_features, eps, momentum, affine)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight_readjust = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias_readjust = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.weight_readjust.data.fill_(0)
        self.bias_readjust.data.fill_(-1)
        self.weight.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        attention = self.sigmoid(self.avg(input) * self.weight_readjust + self.bias_readjust)
        bn_w = self.weight * attention

        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training, self.momentum, self.eps)
        out_bn = out_bn * bn_w + self.bias

        return out_bn

class BAN2d(_BatchAttNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            Mish(),
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            Mish(),
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class BasicResidualSEBlock(nn.Module):

    expansion = 1
    min_condense_feature_shape = 7

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1, bias=False)
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // r, kernel_size=1),
            Mish(),
            nn.Conv2d(out_channels // r, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.resSE_residual = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, groups=out_channels),
            BAN2d(out_channels),
            Mish(),

            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=out_channels),
            BAN2d(out_channels * self.expansion),
        )

        self.resSE_shortcut = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, stride=2, groups=out_channels),
            BAN2d(out_channels),
        )

        self.activation = Mish()

        self.sa = SpatialAttention()

    def dynamic_adaptive_strategy(self, cfe_input):
        cfe_output = cfe_input
        while cfe_output.shape[-2] != self.min_condense_feature_shape and cfe_output.shape[-2] != 1:
            cfe_residual = self.resSE_residual(cfe_output)
            cfe_shortcut = self.resSE_shortcut(cfe_output)
            cfe_output = cfe_residual + cfe_shortcut
            cfe_output = self.activation(cfe_output)

        return cfe_output


    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        residual = self.bn2(residual)
        cfe_output = self.dynamic_adaptive_strategy(residual)
        squeeze = self.squeeze(cfe_output)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        resf = residual * excitation.expand_as(residual)
        residual_SA = self.sa(resf) * resf

        x = residual_SA + shortcut

        return F.relu(x)

class PCXRNet(nn.Module):

    def __init__(self, block, block_num, num_class=100):
        super().__init__()

        self.in_channels = 64
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_stage(block, block_num[0], 64, 1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2)
        self.stage4 = self._make_stage(block, block_num[3], 512, 2)

        self.linear = nn.Linear(self.in_channels, num_class)

    def forward(self, x):
        x = self.pre(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x


    def _make_stage(self, block, num, out_channels, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        while num - 1:
            layers.append(block(self.in_channels, out_channels, 1))
            num -= 1

        return nn.Sequential(*layers)


def PCXRNet34(**kwargs):
    return PCXRNet(BasicResidualSEBlock, [3, 4, 6, 3], **kwargs)
