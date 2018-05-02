import csv
import re
import numpy as np

import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset

from sklearn.utils import shuffle

from datasets import alphabet
# import alphabet

from collections import Counter

class CharDataset(Dataset):
    def __init__(self, config=None, mode='train', ratio=0.8):
        self.mode = mode
        self.ratio = ratio

        self.alphabet = alphabet.alphabet

        self.KEY_LABEL = config['key_label']
        self.KEY_TEXT = config['key_text']

        self.need_regex = config['regex']

        self.data_path = config['path']
        self.lowercase = config['lowercase']

        self.table = config['table']

        self.feature_length = config['length']
        self.feature_size = len(self.alphabet)
        self.class_size = len(self.table)

        self.label = []
        self.text = []

        self.load()

        # self.text, self.label = shuffle(self.text, self.label, random_state=1118)
        self.text, self.label = shuffle(self.text, self.label)

    def __len__(self):
        if self.mode == 'train':
            return int(len(self.label) * self.ratio)
        else:
            return int(len(self.label) * (1.0 - self.ratio))

    def __getitem__(self, idx):
        if self.mode != 'train':
            idx += int(len(self.label) * self.ratio)

        return self.quantization(idx)

    def load(self):
        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')

            next(reader, None)

            for index, row in enumerate(reader):
                self.label.append(row[self.KEY_LABEL])
                self.text.append(row[self.KEY_TEXT])

    def quantization(self, idx):
        sentence = self.text[idx]
        if self.need_regex:
            sentence = re.sub(r'^https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'[^a-zA-Z]', ' ', sentence, flags=re.MULTILINE)

        # print(self.label[idx])
        # print(sentence)

        x = torch.zeros((self.feature_size, self.feature_length))

        # sequence = self.text[idx]

        for i, c in enumerate(sentence):
            if i >= self.feature_length:
                break

            if self.alphabet.find(c) == -1:
                continue

            x[self.alphabet.find(c), i] = 1

        return {'feature':x, 'target': self.table[self.label[idx]]}

    def weight(self):
        return Counter(self.label)

if __name__ == '__main__':

    path = '../../data/uci-news-aggregator.csv'

    table = {
        'b': 0,
        't': 1,
        'e': 2,
        'm': 3,
    }

    dataset_config = {
        'lowercase'     : True,
        'length'        : 50,
        
        'path'          : path,
        'table'         : table,
        'key_label'     : 4,
        'key_text'      : 1,
        'regex'         : False,
    }

    train_dataset = CharDataset(dataset_config)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1)

    print(train_dataset.weight())
