# Deep Residual Learning (ResNet) on CIFAR-10 with Chainer
The training scripts in this repository is based on [this paper](https://arxiv.org/abs/1512.03385).

## Model
* The inputs are 32x32 images
* The first layer is 3x3 convolution
* The next layers are 6n layers with 3x3 convolutions on the feature maps of sizes {32, 16, 8} respectively,
  with 2n layers for each feature map size and the numbers of filters are {16, 32, 64} respectively
* The network ends with a global average pooling, a 10-way fully-connected layer and softmax
* The batch normalization right after each convolution
* The subsampling is performed by convolutions with a stride of 2
* When the dimensions increase, we use 1x1 convolution for the shortcut (Option B in the paper)

## Hyperparameters
* Optimization algorithm is Momentum SGD with rate of 0.9
* Learning rate is started from 0.1 and divide it by 10 at 32k and 48k iterations
* Weight decay is rate of 0.0001
* Mini-batch size is 256
* Weight initialization is scaled Gaussian distribution (refer to [this paper](https://arxiv.org/abs/1502.01852))

## Data Preprocessing and Augumentation
* Per-pixel mean subtraction
* Scaling to the data values between 0 and 1
* Padding 4 pixels on each size
* Randomly cropping 32x32
* Randomly flipping horizontally

## Requirement
* Chainer 2.0.0+

## Start Training
```
$ wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
$ tar zxvf cifar-10-python.tar.gz
$ python train_cifar10.py --gpu 0
```

## Training Curve Example
![Training Curve Example](https://user-images.githubusercontent.com/12968029/29004254-88e3885c-7aff-11e7-8abc-7d1ff645fd2b.png)
