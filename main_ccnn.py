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
import config_char

parser = argparse.ArgumentParser(description='DL Final Project.')

def run(args):
    # data loader
    train_set = dataset.TextDataset(config=config_char.dataset_config, mode='train')
    train_loader = DataLoader(train_set, batch_size=config_char.train_config['batch'], shuffle=True, num_workers=1)

    eval_set = dataset.TextDataset(config=config_char.dataset_config, mode='eval')
    eval_loader = DataLoader(eval_set, batch_size=config_char.train_config['batch'], shuffle=True, num_workers=1)

    # model
    net = model.CharCNN(config=config_char.model_config)
    # net = model.NaiveNN(config=config_char.model_config)

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    # criterion
    criterion = nn.NLLLoss()

    # train
    for epoch in range(config_char.train_config['epochs']):
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
            loss = criterion(output, target)

            # 4. backward
            loss.backward()

            # 5. optimize
            optimizer.step()

            running_loss += loss.data.item()

        tbar.close()

        print(running_loss)

        # Testing
        print('Testing')

        positive = 0
        negative = 0

        trump = 0
        hillary = 0

        vbar = tqdm.tqdm(total=len(eval_loader))

        for batch_idx, sample in enumerate(eval_loader):
        
            vbar.update(1)

            feature, target = sample['feature'], sample['target']
            feature, target = Variable(feature).float(), Variable(target).long()

            output = net(feature)

            _, index = output.max(1)

            # print(torch.sum(index).data[0], torch.sum(target).data[0])
            # print(torch.sum(index == target), torch.sum(index != target))

            positive += (torch.sum(index == target)).data.item()
            negative += (torch.sum(index != target)).data.item()

            hillary += (torch.sum(1 == target)).data.item()
            trump += (torch.sum(0 == target)).data.item()

            # print(positive, negative)

        vbar.close()

        print('acc: ', positive / (positive + negative), 'positive: ', positive, 'negative: ', negative)
        print('hillary: ', hillary, 'trump: ', trump)


if __name__ == '__main__':
    args = parser.parse_args()

    run(args)