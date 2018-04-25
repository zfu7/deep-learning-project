import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import argparse
import tqdm

import dataset
import model
import configuration

parser = argparse.ArgumentParser(description='DL Final Project.')

def run(args):
    # data loader
    train_set = dataset.TextDataset(config=configuration.dataset_config, mode='train')
    train_loader = DataLoader(train_set, batch_size=configuration.train_config['batch'], shuffle=True, num_workers=1)

    # model
    net = model.CharCNN(config=configuration.model_config)

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=configuration.train_config['lr'])

    # criterion
    criterion = nn.CrossEntropyLoss()

    # train
    for epoch in range(configuration.train_config['epochs']):
        # Training
        print('Training')

        running_loss = 0

        tbar = tqdm.tqdm(total=len(train_loader)) # hard code

        for batch_idx, sample in enumerate(train_loader):
            
            tbar.update(1)

            feature, target = sample['feature'], sample['target']
            feature, target = Variable(feature).float(), Variable(target).long()
            
            optimizer.zero_grad()

            # 2. forward
            output = net(feature)
            
            # 3. loss
            loss = F.nll_loss(output, target)

            # 4. backward
            loss.backward()

            # 5. optimize
            optimizer.step()

            running_loss += loss.data[0]

        tbar.close()

        print(running_loss)
        
        # Testing
        print('Testing')

        positive = 0
        negative = 0

        vbar = tqdm.tqdm(total=len(train_loader))

        for batch_idx, sample in enumerate(train_loader):
        
            vbar.update(1)

            feature, target = sample['feature'], sample['target']
            feature, target = Variable(feature).float(), Variable(target).long()

            output = net(feature)

            _, index = output.max(1)

            # print(index.size(), target.size())
            # print(torch.sum(index == target), torch.sum(index != target))

            positive += (torch.sum(index == target)).data[0]
            negative += (torch.sum(index != target)).data[0]

            # print(positive, negative)

        vbar.close()

        print(positive, negative)

if __name__ == '__main__':
    args = parser.parse_args()

    run(args)