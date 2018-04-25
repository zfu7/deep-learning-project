import csv
import torch

from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd

import alphabet

class TextDataset(Dataset):
    def __init__(self, config=None, mode='train'):
        self.mode = mode

        self.data_path = config['path']
        self.lowercase = config['lowercase']

        self.alphabet = alphabet.alphabet
        self.table = alphabet.table

        self.feature_length = config['length']
        self.feature_size = len(self.alphabet)
        self.class_size = len(self.table)

        self.label = []
        self.text = []

        self.load()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.quantization(idx)

    def load(self):
        KEY_LABEL = 1
        KEY_TEXT = 2

        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')

            next(reader, None)

            for index, row in enumerate(reader):

                # if row[KEY_LABEL] == "realDonaldTrump":
                #     self.label.append(0)
                # if row[KEY_LABEL] == "HillaryClinton":
                #     self.label.append(1)

                self.label.append(row[KEY_LABEL])
                self.text.append(row[KEY_TEXT].lower()[:row[KEY_TEXT].find("https://")-1])

    def quantization(self, idx):
        x = torch.zeros((self.feature_size, self.feature_length))
        y = torch.zeros(self.class_size)

        sequence = self.text[idx]

        for idx, c in enumerate(sequence):
            if idx >= self.feature_length:
                break

            if self.alphabet.find(c) == -1:
                continue
            x[self.alphabet.find(c), idx] = 1

        y[self.table[self.label[idx]]] = 1

        return {'feature':x, 'target': self.table[self.label[idx]]}

    def weight(self):
        positive = 0

        for y in self.label:
            positive += self.table[y]

        return positive

if __name__ == '__main__':
    
    label_data_path = '../data/tweets.csv'

    dataset_config = {
        'path': label_data_path,
        'lowercase': True,
        'length': 50,
        'class_size': 2
    }

    train_dataset = TextDataset(dataset_config)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1)

    print(train_dataset.weight())