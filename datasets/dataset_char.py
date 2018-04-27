import csv
import torch

from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd

from sklearn.utils import shuffle

from datasets import alphabet

class CharDataset(Dataset):
    def __init__(self, config=None, mode='train', ratio=0.8):
        self.mode = mode
        self.ratio = ratio

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
        KEY_LABEL = 1
        KEY_TEXT = 2

        with open(self.data_path, 'r') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')

            next(reader, None)

            for index, row in enumerate(reader):

                self.label.append(row[KEY_LABEL])
                self.text.append(row[KEY_TEXT].lower()[:row[KEY_TEXT].find("https://")-1])

    def quantization(self, idx):
        x = torch.zeros((self.feature_size, self.feature_length))

        sequence = self.text[idx]

        for i, c in enumerate(sequence):
            if i >= self.feature_length:
                break

            if self.alphabet.find(c) == -1:
                continue

            x[self.alphabet.find(c), i] = 1

        if self.label[idx] == 'realDonaldTrump':
            return {'feature': x, 'target': 0}
        else:
            return {'feature': x, 'target': 1}

        # return {'feature':x, 'target': self.table[self.label[idx]]}

    def weight(self):
        trump = 0
        hillary = 0

        for idx in range(len(self.label)):
            if self.label[idx] == 'realDonaldTrump':
                trump += 1
            else:
                hillary += 1

        return trump, hillary

if __name__ == '__main__':
    
    pass

    # label_data_path = '../../data/tweets.csv'

    # dataset_config = {
    #     'path': label_data_path,
    #     'lowercase': True,
    #     'length': 100,
    #     'class_size': 2
    # }

    # train_dataset = CharDataset(dataset_config)
    # train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1)

    # print(train_dataset.weight())
