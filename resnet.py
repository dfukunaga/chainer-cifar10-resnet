# Author: dfukunaga (https://github.com/dfukunaga/chainer-cifar10-resnet)

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers


class ResUnitA(chainer.Chain):
    def __init__(self, in_size, out_size, stride):
        super(ResUnitA, self).__init__()
        w = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, out_size, 3, stride, 1, initialW=w, nobias=True)
            self.bn1 = L.BatchNormalization(out_size)
            self.conv2 = L.Convolution2D(
                out_size, out_size, 3, 1, 1, initialW=w, nobias=True)
            self.bn2 = L.BatchNormalization(out_size)
            self.conv3 = L.Convolution2D(
                in_size, out_size, 1, stride, 0, initialW=w, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = self.bn2(self.conv2(h1))
        h2 = self.bn3(self.conv3(x))
        return F.relu(h1 + h2)


class ResUnitB(chainer.Chain):
    def __init__(self, in_size):
        super(ResUnitB, self).__init__()
        w = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, in_size, 3, 1, 1, initialW=w, nobias=True)
            self.bn1 = L.BatchNormalization(in_size)
            self.conv2 = L.Convolution2D(
                in_size, in_size, 3, 1, 1, initialW=w, nobias=True)
            self.bn2 = L.BatchNormalization(in_size)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(x))
        return F.relu(h + x)


class ResBlock(chainer.ChainList):
    def __init__(self, in_size, out_size, stride, layer):
        super(ResBlock, self).__init__()
        self.add_link(ResUnitA(in_size, out_size, stride))
        for i in range(layer - 1):
            self.add_link(ResUnitB(out_size))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class ResNet(chainer.Chain):
    def __init__(self, n=5):
        super(ResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 16, 3, 1, 1, initialW=initializers.HeNormal(), nobias=True)
            self.bn1 = L.BatchNormalization(16)
            self.res2 = ResBlock(16, 16, 1, n)
            self.res3 = ResBlock(16, 32, 2, n)
            self.res4 = ResBlock(32, 64, 2, n)
            self.fc = L.Linear(64, 10)

    def __call__(self, x, t):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = F.average_pooling_2d(h, 8, stride=1)
        h = self.fc(h)

        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss
