"""
Based on code from https://github.com/YisenWang/dynamic_adv_training
"""
from __future__ import print_function
import h5py
import math
import parse
import torch
import torch.nn as nn
import onnx
import numpy as np
import tensorflow.compat.v1 as tf
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.merge import add
from keras import backend as K
BN_AXIS = 3
torch.manual_seed(0)


class DefenseDynamic(torch.nn.Module):
    tf_ckpt = 'checkpoints/Dynamic/advs_cifar-10_ce_0.031.hdf5'
    pt_path = 'checkpoints/Dynamic/advs_cifar-10_ce_0.031.pt'

    def __init__(self, load_tf=False):
        super(DefenseDynamic, self).__init__()
        self.base_model = WideResNet(depth=34, widen_factor=10, num_classes=10)
        if load_tf:
            self.load_tf_weights()
        else:
            self.load_pt_weights()
        self.base_model.eval()
        self.eval()

    def forward(self, x):
        return self.base_model(x)

    def _get_all_conv2d(self):
        l = []
        for n, m in self.base_model.named_modules():
            if isinstance(m, nn.Conv2d):
                l.append((n, m))
        return l

    def _get_all_bns(self):
        l = []
        for n, m in self.base_model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                l.append((n, m))
        return l

    def _convert_conv_weights(self, w):
        assert(len(w.shape) == 4)
        w = torch.FloatTensor(w)
        w = w.permute((3, 2, 0, 1))
        return w

    def _convert_fc_weights(self, w):
        assert(len(w.shape) == 2)
        w = torch.FloatTensor(w)
        w = w.permute((1, 0))
        return w

    def load_tf_weights(self):
        # Load PyTorch modules
        self.pt_conv2d_list = self._get_all_conv2d()
        self.pt_bn_list = self._get_all_bns()

        # Load Keras model
        k_in = Input(shape=(32, 32, 3))
        kmodel = KWideResNet(depth=5, input_tensor=k_in)
        model = Model(inputs=k_in, outputs=kmodel)
        model.load_weights(self.tf_ckpt)
        print(model.summary())

        # Load Weights to PyTorch Model
        conv_idx = 0
        bn_idx = 0
        for layer in model.layers:
            layer_cfg = layer.get_config()
            layer_w = layer.get_weights()
            name = layer_cfg['name']
            if 'conv2d' in name:
                m = self.pt_conv2d_list[conv_idx]
                conv_idx += 1
                w = layer_w[0]
                b = layer_w[1]
                w = self._convert_conv_weights(w)
                b = torch.FloatTensor(b)
                assert(w.shape == m[1].weight.data.shape)
                assert(b.shape == m[1].bias.data.shape)
                m[1].weight.data = w
                m[1].bias.data = b
                print(name, m[0], 'Loaded')
            elif 'batch_normalization' in name:
                m = self.pt_bn_list[bn_idx]
                bn_idx += 1
                w = layer_w[0]
                b = layer_w[1]
                mean = layer_w[2]
                var = layer_w[3]
                w = torch.FloatTensor(w)
                b = torch.FloatTensor(b)
                mean = torch.FloatTensor(mean)
                var = torch.FloatTensor(var)
                assert(w.shape == m[1].weight.data.shape)
                assert(b.shape == m[1].bias.data.shape)
                assert(mean.shape == m[1].running_mean.shape)
                assert(var.shape == m[1].running_var.shape)
                m[1].weight.data = w
                m[1].bias.data = b
                m[1].running_mean = mean
                m[1].running_var = var
                print(name, m[0], 'Loaded')
            elif 'logits' in name:
                m = ('fc', self.base_model.fc)
                w = layer_w[0]
                b = layer_w[1]
                w = self._convert_fc_weights(w)
                b = torch.FloatTensor(b)
                assert(w.shape == m[1].weight.data.shape)
                assert(b.shape == m[1].bias.data.shape)
                m[1].weight.data = w
                m[1].bias.data = b
                print(name, m[0], 'Loaded')

        # model = Model(inputs=k_in, outputs=model.get_layer('conv2d_14').output)
        # x = np.array([np.zeros((32, 32, 3), dtype=np.float)])
        # self.tf_out = model.predict(x)
        #
        # self.base_model.eval()
        # print(self.tf_out, self.tf_out.shape)
        # logits = self.forward(torch.zeros(1, 3, 32, 32))
        # self.base_model.pt_out = logits.detach().numpy()
        # print('='*10)
        # print(self.base_model.pt_out, self.base_model.pt_out.shape)
        # print('='*10)
        # diff = np.abs(self.tf_out - self.base_model.pt_out)
        # print(diff, diff.mean())
        # print('='*10)
        # print(logits)
        torch.save(self.base_model.state_dict(), self.pt_path)
        return

    def load_pt_weights(self):
        self.base_model.load_state_dict(torch.load(self.pt_path))


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, l_idx=0, dropRate=0.0,):
        super(BasicBlock, self).__init__()
        self.relu1 = nn.ReLU(inplace=False)
        # https://stackoverflow.com/questions/61422046/resnet-model-of-pytorch
        # -and-tensorflow-give-different-results-when-stride-2
        if stride == 1:
            self.pad1 = None
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                   stride=stride, padding=1, bias=True)
        elif stride == 2:
            self.pad1 = nn.ZeroPad2d((0, 1, 0, 1))  # 0,1,0,1
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                   stride=stride, padding=0, bias=True)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0,
            bias=True) or None
        self.bn2 = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=True)
        self.droprate = dropRate
        self.l_idx = l_idx

    def forward(self, x):
        out = self.relu1(x)
        if self.pad1 is not None:
            o_x = x
        else:
            o_x = out
        if self.pad1 is not None:
            out = self.conv1(self.pad1(out))
        else:
            out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        if self.equalInOut:
            out = torch.add(x, out)
        else:
            o_x = self.convShortcut(o_x)
            out = torch.add(o_x, out)
        return out


class NetworkBlock(nn.Module):
    def __init__(self,
                 nb_layers,
                 in_planes,
                 out_planes,
                 block,
                 stride,
                 dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                      stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,
                    dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes,
                      i == 0 and stride or 1, l_idx=i, dropRate=dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [
            16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor
        ]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.latent = False

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3,
                               stride=1, padding=1, bias=True)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1,
                                   dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2,
                                   dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2,
                                   dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        # Expose features in the final residual blocks
        sequential = self.block3.layer
        features = []
        for s in sequential:
            out = s(out)
            features.append(out)
        out = self.relu(self.bn1(out))
        out = out.mean(dim=[2, 3])
        pooled = out
        out = self.fc(out)
        if self.latent:
            features += [pooled, out]
            return features
        return out


def KWideResNet(depth, n_class=10, input_tensor=None):
    """
    10 times wider than ResNet.
    total number of layers: 2 + 6 * depth
    :param depth:
    :param n_class:
    :param input_tensor:
    :return: sequence of layers until the logits
    """

    num_conv = 3
    decay = 2e-3

    # 1 conv + BN + relu
    filters = 16
    b = Conv2D(filters=filters, kernel_size=(num_conv, num_conv),
               kernel_initializer="he_normal", padding="same",
               kernel_regularizer=l2(decay), bias_regularizer=l2(0))(input_tensor)
    b = Activation("relu")(b)

    filters *= 10  # wide

    # 1 res, no striding
    b = residual(num_conv, filters, decay, first=True)(b)  # 2 layers inside
    for _ in np.arange(1, depth):  # start from 1 => 2 * depth in total
        b = residual(num_conv, filters, decay)(b)

    filters *= 2

    # 2 res, with striding
    b = residual(num_conv, filters, decay, more_filters=True)(b)
    for _ in np.arange(1, depth):
        b = residual(num_conv, filters, decay)(b)

    filters *= 2

    # 3 res, with striding
    b = residual(num_conv, filters, decay, more_filters=True)(b)
    for _ in np.arange(1, depth):
        b = residual(num_conv, filters, decay)(b)

    b = BatchNormalization(axis=BN_AXIS)(b)
    b = Activation("relu")(b)

    b = AveragePooling2D(pool_size=(8, 8), strides=(1, 1),
                         padding="valid")(b)

    b = Flatten(name='features')(b)

    dense = Dense(units=n_class, kernel_initializer="he_normal",
                  kernel_regularizer=l2(decay), bias_regularizer=l2(0), name='logits')(b)

    return dense


def residual(num_conv, filters, decay, more_filters=False, first=False):
    def f(input):
        # in_channel = input._keras_shape[1]
        out_channel = filters

        if more_filters and not first:
            # out_channel = in_channel * 2
            stride = 2
        else:
            # out_channel = in_channel
            stride = 1

        if not first:
            b = BatchNormalization(axis=BN_AXIS)(input)
            b = Activation("relu")(b)
            b = Activation("relu")(input)
        else:
            b = input

        b = Conv2D(filters=out_channel,
                   kernel_size=(num_conv, num_conv),
                   strides=(stride, stride),
                   kernel_initializer="he_normal", padding="same",
                   kernel_regularizer=l2(decay), bias_regularizer=l2(0))(b)
        b = BatchNormalization(axis=BN_AXIS)(b)
        b = Activation("relu")(b)
        res = Conv2D(filters=out_channel,
                     kernel_size=(num_conv, num_conv),
                     kernel_initializer="he_normal", padding="same",
                     kernel_regularizer=l2(decay), bias_regularizer=l2(0))(b)

        # check and match number of filter for the shortcut
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(res)
        if not input_shape[3] == residual_shape[3]:
            stride_width = int(round(input_shape[1] / residual_shape[1]))
            stride_height = int(round(input_shape[2] / residual_shape[2]))

            input = Conv2D(filters=residual_shape[3], kernel_size=(1, 1),
                           strides=(stride_width, stride_height),
                           kernel_initializer="he_normal",
                           padding="valid", kernel_regularizer=l2(decay))(input)

        return add([input, res])

    return f


if __name__ == '__main__':
    test = DefenseDynamic(load_tf=True)
