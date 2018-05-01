import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='DL Final Project.')

parser.add_argument('--type', type=str, help='type for (char, word)')
parser.add_argument('--model', type=str, help='type for (char, word)')
parser.add_argument('--dataset', type=str, help='dataset for (tweets, news)')

# args = None

def plot_acc(train_acc, val_acc):
    train, = plt.plot(train_acc, label='Training')
    val, = plt.plot(val_acc, label='Validation')

    plt.legend(handles=[train, val], loc='lower left', ncol=3, fontsize=8)
    plt.title('Training and Validation Accuracy Plot: ' + args.model)

    plt.show()

def plot_loss(loss):
    plt.plot(loss, label='Training')
    plt.title('Training Loss Plot: ' + args.model)
    plt.show()

def load_data(filename):
    f = open(filename, 'rb')
    data = np.load(f)
    f.close()
    return data

if __name__ == '__main__':
    global args
    args = parser.parse_args()

    loss = 'results/loss_' + args.type + '_' + args.model + '_' + args.dataset
    train_acc = 'results/acc_' + args.type + '_' + args.model + '_train_' + args.dataset
    val_acc = 'results/acc_' + args.type + '_' + args.model + '_val_' + args.dataset

    loss = load_data(loss)
    train_acc = load_data(train_acc)
    val_acc = load_data(val_acc)

    plot_acc(train_acc, val_acc)
    plot_loss(loss)