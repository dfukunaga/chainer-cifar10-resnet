#!/usr/bin/env python

import argparse
import pickle
import random

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

import resnet


def load_cifar10_dataset(path, test=False):
    if test:
        files = ['test_batch']
        data = np.zeros((10000, 3, 32, 32), dtype=np.float32)
        labels = np.zeros(10000, dtype=np.int32)
    else:
        files = ['data_batch_{}'.format(i) for i in range(1, 6)]
        data = np.zeros((50000, 3, 32, 32), dtype=np.float32)
        labels = np.zeros(50000, dtype=np.int32)

    for i, file in enumerate(files):
        with open('{}/{}'.format(path, file), 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        data[i*10000:(i+1)*10000] = dict[b'data'].reshape(10000, 3, 32, 32)
        labels[i*10000:(i+1)*10000] = dict[b'labels']

    return data, labels


class Cifar10Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, data, labels, train=True, mean=0):
        self.data = data
        self.labels = labels
        self.train = train

        # Per-pixel mean subtraction
        self.data = self.data - mean

        # Scale to [0, 1]
        self.data = self.data * (1.0 / 255.0)

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        # This method applies following preprocesses (only training dataset):
        #     - Padding 4 pixels on each size
        #     - 32x32 crop randomly
        #     - Horizontal flip randomly
        image = self.data[i]
        label = self.labels[i]

        if self.train:
            # Padding
            image = np.pad(image, ((0, 0), (4, 4), (4, 4)), 'constant')

            # Crop
            top = random.randint(0, 7)
            left = random.randint(0, 7)
            image = image[:, top:top+32, left:left+32]

            # Flip
            if random.randint(0, 1):
                image = image[:, :, ::-1]

        return image, label


def main():
    parser = argparse.ArgumentParser(description='Learning CIFAR-10 using ResNet')
    parser.add_argument('--batchsize', type=int, default=256, help='Leaning minibatch size')
    parser.add_argument('--epoch', type=int, default=365, help='Number of epochs to train')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU id (-1 indicates CPU)')
    parser.add_argument('--loaders', type=int, default=1, help='Number of data loading processes')
    parser.add_argument('--out', default='./result', help='Path of output directory')
    parser.add_argument('--path', default='./cifar-10-batches-py', help='Path of dataset files')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    print('==========================================')
    print('Num Minibatch-size: {}'.format(args.batchsize))
    print('Num Epoch: {}'.format(args.epoch))
    print('==========================================')

    # Model
    model = resnet.ResNet()
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Dataset
    data, labels = load_cifar10_dataset(args.path)
    mean = np.mean(data, axis=0)
    train = Cifar10Dataset(data[0:45000], labels[0:45000], train=True, mean=mean)
    val = Cifar10Dataset(data[45000:50000], labels[45000:50000], train=False, mean=mean)

    # Iterators
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize, n_processes=args.loaders)
    val_iter = chainer.iterators.MultiprocessIterator(val, args.batchsize, repeat=False, shuffle=False, n_processes=args.loaders)

    # Optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.1, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    # Trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    # Validation
    val_interval = (10, 'iteration') if args.test else (1, 'epoch')
    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'), trigger=val_interval)

    # Learning rate decay
    lr_interval = training.triggers.ManualScheduleTrigger([32000, 48000], 'iteration')
    trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=lr_interval)

    # Log
    log_interval = (10, 'iteration') if args.test else (1, 'epoch')
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'lr', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()

if __name__ == '__main__':
    main()
