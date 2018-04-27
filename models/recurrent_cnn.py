# Recurrent Convolutional Neural Networks

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

class RCNN(nn.Module):

    def __init__(self, config=None):
        super(RCNN, self).__init__()

        self.cuda_on = torch.cuda.is_available()

        # configuration
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.context_dim = config['context_dim']
        self.lstm_layers = config['lstm_layers']

        self.label_dim = config['label_dim']

        # essentially it is a bidirectional lstm
        self.bilstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.context_dim, num_layers=self.lstm_layers, dropout=0.5, bidirectional=True)

        # linear layer
        self.feature2hidden = nn.Linear(2 * self.context_dim + self.embedding_dim, self.hidden_dim)
        self.hidden2output = nn.Linear(self.hidden_dim, self.label_dim)

        # log softmax (using nll loss)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        self.batch_size = x.size()[0]

        self.hidden =  self.init_hidden(self.batch_size)

        lstm_out, self.hidden = self.bilstm(x.permute(1,0,2), self.hidden)

        features = torch.cat((lstm_out.permute(1,0,2), x), dim=2)

        outputs = F.tanh(self.feature2hidden(features)).permute(0, 2, 1)

        activations = F.max_pool1d(outputs, kernel_size=outputs.size()[2]).view(self.batch_size, -1)

        return self.log_softmax(self.hidden2output(activations))

    def init_hidden(self, batch_size=None):
        hidden_state = Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.context_dim))
        cell_state   = Variable(torch.zeros(2 * self.lstm_layers, batch_size, self.context_dim))

        if self.cuda_on:
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()
        return (hidden_state, cell_state)

if __name__ == '__main__':
    config = {
        'embedding_dim' : 50,
        'hidden_dim'    : 50,
        'context_dim'   : 50,
        'label_dim'     : 2,
        'lstm_layers'   : 2
    }

    net = RCNN(config)

    x = Variable(torch.rand(4, 1000, 50))

    output = net(x)

    print(output.size())

    print(output)

    _, index = output.max(1)

    print(index)






