import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable


class CharCNN(nn.Module):

    def __init__(self, config=None):
        super(CharCNN, self).__init__()

        self.conv1 = self._build_conv_layer(config['feature_num'], 256, 7, True)
        self.conv2 = self._build_conv_layer(256, 256, 7, True)
        self.conv3 = self._build_conv_layer(256, 256, 3, False)
        self.conv4 = self._build_conv_layer(256, 256, 3, False)
        self.conv5 = self._build_conv_layer(256, 256, 3, False)
        # self.conv6 = self._build_conv_layer(256, 256, 3, True)

        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=config['dropout'])
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=config['dropout'])
        )

        self.fc3 = nn.Linear(1024, config['class_num'])
        self.log_softmax = nn.LogSoftmax(dim=1)

        self._initialize()

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x = self.conv2(x)
        # print(x.size())
        x = self.conv3(x)
        # print(x.size())
        x = self.conv4(x)
        # print(x.size())
        x = self.conv5(x)
        # print(x.size())
        # x = self.conv6(x)
        # print(x.size())

        # collapse
        x = x.view(x.size(0), -1)
        # print(x.size())
        # linear layer
        x = self.fc1(x)
        # print(x.size())
        # linear layer
        x = self.fc2(x)
        # print(x.size())
        # linear layer
        x = self.fc3(x)
        # print(x.size())
        # output layer
        x = self.log_softmax(x)

        return x

    def _build_conv_layer(self, input_channels=None, output_channels=None, kernel=None, max_pooling=False):
        layers = []

        layers.append(nn.Conv1d(input_channels, output_channels, kernel_size=kernel, stride=1))
        layers.append(nn.ReLU(inplace=True))

        if max_pooling:
            layers.append(nn.MaxPool1d(kernel_size=3, stride=3))

        return nn.Sequential(*layers)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()        
        return

if __name__ == '__main__':
    config = {
        'feature_num': 70,
        'class_num':   2,
        'dropout': 0.5
    }
    char_cnn = CharCNN(config)

    paras = sum(p.numel() for p in char_cnn.parameters() if p.requires_grad)
    # print(paras)

    x = Variable(torch.rand(4, 70, 100))

    output = char_cnn(x)

    # print(output.size())
