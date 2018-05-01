import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

class CharVgg(nn.Module):

    def __init__(self, config=None):
        super(CharVgg, self).__init__()

        self.dropout_rate = config['dropout']

        self.conv_layers = []
        self.fc_layers = []

        for idx in range(len(config['conv_layers'])):
            self.conv_layers.append(self._build_conv_layer(config['conv_layers'][idx]))
        self.conv_layers = nn.ModuleList(self.conv_layers)

        for idx in range(len(config['fc_layers'])):
            self.fc_layers.append(self._build_fc_layer(config['fc_layers'][idx]))
        self.fc_layers = nn.ModuleList(self.fc_layers)

        self.log_softmax = nn.LogSoftmax(dim=1)

        self._initialize()

    def forward(self, x):

        for conv in self.conv_layers:
            x = conv(x)

        # collapse
        x = x.view(x.size(0), -1)

        for fc in self.fc_layers:
            x = fc(x)

        return self.log_softmax(x)

    def _build_conv_layer(self, config):
        layers = []

        layers.append(nn.Conv1d(in_channels=config[0], out_channels=config[1], kernel_size=config[2], padding=1))
        layers.append(nn.BatchNorm1d(num_features=config[1]))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv1d(in_channels=config[1], out_channels=config[1], kernel_size=config[2], padding=1))
        layers.append(nn.BatchNorm1d(num_features=config[1]))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def _build_fc_layer(self, config):
        layers = []

        layers.append(nn.Linear(config[0], config[1]))

        if config[2]:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=self.dropout_rate))

        return nn.Sequential(*layers)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()        
        return

class NaiveNN(nn.Module):

    def __init__(self, config=None):
        super(NaiveNN, self).__init__()

        self.fc = nn.Linear(70 * 100, config['class_num'])
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.log_softmax(x)

        return x

if __name__ == '__main__':
    conv_config = [
        [70, 32, 3, True, True],
        [32, 32, 3, True, False],
        [32, 32, 3, True, False],
        [32, 32, 3, True, False],
        [32, 32, 3, True, True],
    ]

    fc_layers = [
        [96,  256, True],
        [256, 256, True],
        [256, 2,   False],
    ]

    config = {
        'dropout'       : 0.5,
        'conv_layers'   : conv_config,
        'fc_layers'     : fc_layers,
    }

    net = CharVgg(config)


    paras = sum(p.numel() for p in net.parameters() if p.requires_grad)

    x = Variable(torch.rand(4, 70, 100))

    output = net(x)

    print(output.size())
