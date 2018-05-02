import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import argparse
import tqdm
import numpy as np

from sklearn.metrics import confusion_matrix

from datasets import dataset_char, dataset_word
from models import char_cnn, recurrent_cnn, char_vgg, char_res
from configs import config_char_cnn, config_char_vgg, config_char_res
from configs import config_word_rcnn
from configs import config_tweets, config_uci_news, config_ag_news

parser = argparse.ArgumentParser(description='DL Final Project.')

parser.add_argument('--type', type=str, help='type for (char, word)')
parser.add_argument('--model', type=str, help='type for (char, word)')
parser.add_argument('--dataset', type=str, help='dataset for (tweets, news)')
parser.add_argument('--mode', type=str, help='train or test', default='train')

n_class = None
table = None

cuda_on = torch.cuda.is_available()

def run(args):
    if args.dataset == 'tweets':
        dataset_config = config_tweets.dataset_config
    elif args.dataset == 'news':
        dataset_config = config_uci_news.dataset_config
    elif args.dataset == 'ag':
        dataset_config = config_ag_news.dataset_config
    elif args.dataset == 'ag_test':
        dataset_config = config_ag_news_test.dataset_config
    else:
        raise ValueError('unknown dataset: ' + args.dataset)

    global n_class
    n_class = len(dataset_config['table'])

    global table
    table = dataset_config['table']

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
        config.model_config['fc_layers'][-1][1] = n_class

    elif args.type == 'word':
        # dataset
        dataset = dataset_word.WordDataset

        if args.model == 'rcnn':
            # configuration
            config = config_word_rcnn   
            # model
            model = recurrent_cnn.RCNN

        
        # adjust output channels size
        config.model_config['label_dim'] = n_class

    else:
        raise ValueError('unknown type type: ' + args.type)

    if args.mode == 'train':

        # data loader
        print('loading dataset')
        train_set = dataset(config=dataset_config, mode='train')
        train_loader = DataLoader(train_set, batch_size=config.train_config['batch'], shuffle=True, num_workers=1)

        eval_set = dataset(config=dataset_config, mode='eval')
        eval_loader = DataLoader(eval_set, batch_size=config.train_config['batch'], shuffle=True, num_workers=1)

        # model
        net = model(config=config.model_config)
        if cuda_on:
            net = net.cuda()

        # optimizer
        optimizer = optim.SGD(net.parameters(), lr=config.train_config['lr'], momentum=config.train_config['momentum'])

        # criterion
        criterion = nn.NLLLoss()

        n_epochs = config.train_config['epochs']

        loss        = np.zeros(n_epochs)
        train_acc   = np.zeros(n_epochs)
        val_acc     = np.zeros(n_epochs)

        # train
        if config.file_config['pretrained']:
            net.load_state_dict(torch.load(config.file_config['model'] + '_' + args.dataset + '.py'))

        for epoch in range(config.train_config['epochs']):
            print('epoch: ', epoch)

            torch.save(net.state_dict(), config.file_config['model'] + '_' + args.dataset + '.py')
            
            loss[epoch]         = (train(train_loader, net, optimizer, criterion))

            train_acc[epoch]    = (validate(train_loader, net, 'train'))
            val_acc[epoch]      =(validate(eval_loader, net, 'val'))

            save_arr(loss, config.file_config['loss'] + '_' + args.dataset)
            save_arr(train_acc, config.file_config['acc'] + '_' + 'train' + '_' + args.dataset)
            save_arr(val_acc, config.file_config['acc'] + '_' + 'val' + '_' + args.dataset)


    elif args.mode == 'test':
        test_set = dataset(config=dataset_config, ratio=1.0)
        test_loader = DataLoader(eval_set, batch_size=config.train_config['batch'], shuffle=True, num_workers=1)

        net = model(config=config.model_config)
        if cuda_on:
            net = net.cuda()

        model_path = config.file_config['model'] + '_' + args.dataset + '.py'

        if cuda_on:
            net.load_state_dict(torch.load(model_path))
        else:
            net.load_state_dict(torch.load(model_path, map_location=lambda storage, location: 'cpu'))
        
        validate(loader, net)

def train(loader, net, optimizer, criterion):
    # Training
    print('Training')

    running_loss = 0

    tbar = tqdm.tqdm(total=len(loader)) # hard code

    batch = 0

    for batch_idx, sample in enumerate(loader):

        tbar.update(1)

        feature, target = sample['feature'], sample['target']
        feature, target = Variable(feature).float(), Variable(target).long()

        if cuda_on:
            feature = feature.cuda()
            target = target.cuda()

        if feature.size() == 1:
            continue
        
        optimizer.zero_grad()

        # 2. forward
        output = net(feature)
        
        # 3. loss
        loss = criterion(output, target)

        # if batch < 100 and args.type == 'word':
        #     batch += 1
        #     continue

        # 4. backward
        loss.backward()

        # 5. optimize
        optimizer.step()

        running_loss += loss.data[0]

    tbar.close()

    print('Training loss: ', running_loss)

    return running_loss


def validate(loader, net, mode=None):
    # validate
    print('Validate')

    vbar = tqdm.tqdm(total=len(loader))

    c_mat = np.zeros((n_class, n_class), dtype=np.integer)

    for batch_idx, sample in enumerate(loader):
    
        vbar.update(1)

        feature, target = sample['feature'], sample['target']
        feature, target = Variable(feature).float(), Variable(target).long()

        if cuda_on:
            feature, target = feature.cuda(), target.cuda()

        if feature.size() == 1:
            continue

        output = net(feature)

        _, index = output.max(1)

        c_mat += confusion_matrix(target.cpu().data.numpy(), 
                                  index.cpu().data.numpy(),
                                  list(range(0, n_class)))

    vbar.close()

    positive = np.trace(c_mat)
    negative = np.sum(c_mat) - positive
    acc = np.float(positive) / np.float(positive + negative)

    print(c_mat)

    print(mode, 'acc: ', acc, 'positive: ', positive, 'negative: ', negative)

    return acc


def save_arr(data, filename):
    f = open(filename, 'wb')
    np.save(f, data)
    f.close()


def load_arr(filename):
    f = open(filename, 'rb')
    data = np.load(f)
    f.close()
    return data


if __name__ == '__main__':
    args = parser.parse_args()

    run(args)
