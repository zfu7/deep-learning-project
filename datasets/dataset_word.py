import csv
import re
import numpy as np

import torch

import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset

from sklearn.utils import shuffle

class WordDataset(Dataset):
    def __init__(self, config=None, mode='train', ratio=0.8):
        self.mode = mode
        self.ratio = ratio

        self.regex = re.compile('[^a-zA-Z]')

        self.embedding = {}
        self.embedding_dim = config['embedding_dim']

        self.KEY_LABEL = config['key_label']
        self.KEY_TEXT = config['key_text']

        self.need_regex = config['regex']

        self.word_size = config['word_size']

        self.table = config['table']

        self.text = []
        self.label = []

        self.load_embedding(config['embedding_path'])
        self.load_dataset(config['path'])

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

        return self.word_to_vec(idx)

    def load_dataset(self, path=None):
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')

            next(reader, None)

            for index, row in enumerate(reader):

                self.label.append(row[self.KEY_LABEL])
                # self.text.append(row[KEY_TEXT].lower()[:row[KEY_TEXT].find("https://")-1])
                self.text.append(row[self.KEY_TEXT].lower())

    def word_to_vec(self, idx):
        sentence = self.text[idx]

        if self.need_regex:
            sentence = re.sub(r'^https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'[^a-zA-Z]', ' ', sentence, flags=re.MULTILINE)

        x = torch.zeros((self.word_size, self.embedding_dim))

        for index, word in enumerate(sentence):
            if index >= self.word_size:
                break

            if word in self.embedding.keys():
                x[index, :] = torch.from_numpy(self.embedding[word])

        return {'feature': x, 'target': self.table[self.label[idx]]}

    def load_embedding(self, path=None):
        with open(path, 'r') as f:

            for line in f:
                split_line = line.split()
                self.embedding[split_line[0]] = np.array([float(val) for val in split_line[1:]])


if __name__ == '__main__':
    dataset = WordDataset

    # tweets
    path = '../data/tweets.csv'

    table = {
        'realDonaldTrump': 0,
        'HillaryClinton': 1
    }

    dataset_config = {
        'lowercase'     : True,
        'length'        : 100,
        
        'path'          : path,

        'table'         : table,

        'key_label'     : 1,
        'key_text'      : 2,

        'regex'         : True,

        'embedding_path': '../glove/glove.twitter.27B.50d.txt',
        'embedding_dim' : 50,
    }

    train_dataset = dataset(dataset_config)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1)

    train_dataset.__getitem__(10)