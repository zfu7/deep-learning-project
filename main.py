import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import argparse
import tqdm

from datasets import dataset_char, dataset_word
from models import char_cnn, recurrent_cnn, char_vgg, char_res
from configs import config_char_cnn, config_char_vgg, config_char_res, config_word
from configs import config_tweets, config_uci_news

parser = argparse.ArgumentParser(description='DL Final Project.')

parser.add_argument('--type', type=str, help='type for (char, word)')
parser.add_argument('--model', type=str, help='type for (char, word)')
parser.add_argument('--dataset', type=str, help='dataset for (tweets, news)')

def run(args):
    if args.dataset == 'tweets':
        dataset_config = config_tweets.dataset_config
    elif args.dataset == 'news':
        dataset_config = config_uci_news.dataset_config
    else:
        raise ValueError('unknown dataset: ' + args.dataset)

    if args.type == 'char':
        # dataset
        dataset = dataset_char.CharDataset
        # configuration
        if args.model == 'cnn':
            config = config_char_cnn
            model = char_cnn.CharCNN
        elif args.model == 'vgg':
            config = config_char_vgg
            model = char_vgg.CharVgg
        elif args.model == 'res':
            config = config_char_res
            model = char_res.CharRes

        # adjust output channels size
        config.model_config['fc_layers'][-1][1] = len(dataset_config['table'])

    elif args.type == 'word':
        # dataset
        dataset = dataset_word.WordDataset
        # configuration
        config = config_word
        # model
        model = recurrent_cnn.RCNN
        
        # adjust output channels size
        config.model_config['label_dim'] = len(dataset_config['table'])

    else:
        raise ValueError('unknown type type: ' + args.type)

    # data loader
    print('loading dataset')
    train_set = dataset(config=dataset_config, mode='train')
    train_loader = DataLoader(train_set, batch_size=config.train_config['batch'], shuffle=True, num_workers=1)

    eval_set = dataset(config=dataset_config, mode='eval')
    eval_loader = DataLoader(eval_set, batch_size=config.train_config['batch'], shuffle=True, num_workers=1)

    # model
    net = model(config=config.model_config)

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=config.train_config['lr'], momentum=config.train_config['momentum'])

    # criterion
    criterion = nn.NLLLoss()

    # train
    for epoch in range(config.train_config['epochs']):
        # Training
        print('Training')

        running_loss = 0

        tbar = tqdm.tqdm(total=len(train_loader)) # hard code

        batch = 0

        for batch_idx, sample in enumerate(train_loader):

            tbar.update(1)

            feature, target = sample['feature'], sample['target']
            feature, target = Variable(feature).float(), Variable(target).long()

            if feature.size() == 1:
                continue
            
            optimizer.zero_grad()

            # 2. forward
            output = net(feature)
            
            # 3. loss
            loss = criterion(output, target)

            if batch < 100 and args.type == 'word':
                batch += 1
                continue

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

            if feature.size() == 1:
                continue

            output = net(feature)

            _, index = output.max(1)

            positive += (torch.sum(index == target)).data.item()
            negative += (torch.sum(index != target)).data.item()

            # hillary += (torch.sum(1 == target)).data.item()
            # trump += (torch.sum(0 == target)).data.item()

            # print(positive, negative)

        vbar.close()

        print('acc: ', positive / (positive + negative), 'positive: ', positive, 'negative: ', negative)
        # print('hillary: ', hillary, 'trump: ', trump)


if __name__ == '__main__':
    args = parser.parse_args()

    run(args)