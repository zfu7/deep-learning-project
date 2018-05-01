import numpy as np

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import torch

from torch.autograd import Variable

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_layer=None, stride=1):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.skip_layer = skip_layer

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv1_bn = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.conv2_bn(self.conv2(x))

        x += residual

        return F.relu(x)

class CharRes(nn.Module):
    def __init__(self, config):
        super(CharRes, self).__init__()

        self.conv = nn.Conv1d(config['input'], config['residual'][0][0], kernel_size=3, stride=1, padding=1)
        self.bn   = nn.BatchNorm1d(4)

        self.res_layers = []
        for idx in range(len(config['residual'])):
            self.res_layers.append(self._construct_layers(config['residual'][idx][0], 
                                                          config['residual'][idx][1],
                                                          config['residual'][idx][2],
                                                          config['residual'][idx][3]))
        self.res_layers = nn.ModuleList(self.res_layers)

        self.avgpool = nn.AvgPool1d(kernel_size=7)

        self.fc = nn.Linear(config['fc_layers'][0][0], config['fc_layers'][0][1])

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _construct_layers(self, in_channels, out_channels, n_blocks, stride=1):
        layers = []
        if in_channels != out_channels:
            skip_layer = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                                       nn.BatchNorm1d(out_channels))
            layers.append(BasicBlock(in_channels, out_channels, skip_layer=skip_layer, stride=stride))
        else:   
            layers.append(BasicBlock(in_channels, out_channels, stride=stride))
        for idx in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv0_bn(self.conv0(x))), kernel_size=3, stride=2, padding=1)
        x = F.relu(self.bn(self.conv(x)))

        for res_layer in self.res_layers:
            x = res_layer(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    # input channels, output channels, kernel size, batch normalization, max pooling

    res_config = [
        [4,  8,  2, 1],
        [8,  16, 2, 2],
        [16, 32, 2, 2],
        [32, 64, 2, 1],
    ]

    config = {
        'input'         : 70,
        'residual'      : res_config,
    }

    net = CharRes(config)
    # print(net)

    paras = sum(p.numel() for p in net.parameters() if p.requires_grad)

    x = Variable(torch.rand(4, 70, 100))

    output = net(x)

    print(output.size())
