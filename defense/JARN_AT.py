"""
Based on code from https://github.com/alvinchangw/JARN_ICLR2020
"""
import os
import math
import parse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow.compat.v1 as tf
import torchvision.transforms as transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def get_tf_key(torch_name):
    if torch_name == 'conv1.weight':
        return "classifier/input/init_conv/DW:0"
    elif torch_name.startswith('block') and 'bn' in torch_name:
        f = "block{:d}.layer.{:d}.bn{:d}.{}"
        parsed = parse.parse(f, torch_name)
        b = parsed[0]
        l = parsed[1]
        n = parsed[2]
        t = parsed[3]
        if t == 'weight':
            t = 'gamma'
        elif t == 'bias':
            t = 'beta'
        if b == 1 and l == 0 and n == 1:
            return "classifier/unit_1_0/shared_activation/BatchNorm/{}:0".format(t)
        elif n == 1:
            return "classifier/unit_{:d}_{:d}/residual_only_activation/BatchNorm/{}:0".format(b, l, t)
        elif n == 2:
            return "classifier/unit_{:d}_{:d}/sub2/BatchNorm/{}:0".format(b, l, t)
    elif torch_name.startswith('block') and 'conv' in torch_name:
        # print(torch_name)
        f = "block{:d}.layer.{:d}.conv{:d}.weight"
        parsed = parse.parse(f, torch_name)
        b = parsed[0]
        l = parsed[1]
        n = parsed[2]
        return "classifier/unit_{:d}_{:d}/sub{:d}/conv{:d}/DW:0".format(b, l, n, n)
    elif torch_name == 'bn1.weight':
        return 'classifier/unit_last/BatchNorm/gamma:0'
    elif torch_name == 'bn1.bias':
        return 'classifier/unit_last/BatchNorm/beta:0'
    elif torch_name == 'fc.weight':
        return "classifier/logit/DW:0"
    elif torch_name == 'fc.bias':
        return "classifier/logit/biases:0"


def get_tf_bn_mean_var(torch_name):
    m, b = None, None
    if torch_name.startswith('block') and 'bn' in torch_name:
        f = "block{:d}.layer.{:d}.bn{:d}"
        parsed = parse.parse(f, torch_name)
        b = parsed[0]
        l = parsed[1]
        n = parsed[2]

        if b == 1 and l == 0 and n == 1:
            m = "classifier/unit_1_0/shared_activation/BatchNorm/moving_mean:0"
            v = "classifier/unit_1_0/shared_activation/BatchNorm/moving_variance:0"
        elif n == 1:
            m = "classifier/unit_{:d}_{:d}/residual_only_activation/BatchNorm/moving_mean:0".format(b, l)
            v = "classifier/unit_{:d}_{:d}/residual_only_activation/BatchNorm/moving_variance:0".format(b, l)
        elif n == 2:
            m = "classifier/unit_{:d}_{:d}/sub2/BatchNorm/moving_mean:0".format(b, l)
            v = "classifier/unit_{:d}_{:d}/sub2/BatchNorm/moving_variance:0".format(b, l)
    elif torch_name == 'bn1':
        m = 'classifier/unit_last/BatchNorm/moving_mean:0'
        v = 'classifier/unit_last/BatchNorm/moving_variance:0'
    return m, v


def batch_per_image_standardization(imgs):
    # replicate tf.image.per_image_standardization, but in batch
    assert imgs.ndimension() == 4
    mean = imgs.view(imgs.shape[0], -1).mean(dim=1).view(
        imgs.shape[0], 1, 1, 1)
    return (imgs - mean) / batch_adjusted_stddev(imgs)
    # return F.normalize(imgs, dim=[1, 2, 3], p=2)


def batch_adjusted_stddev(imgs):
    # for batch_per_image_standardization
    std = imgs.view(imgs.shape[0], -1).std(dim=1).view(imgs.shape[0], 1, 1, 1)
    std_min = 1. / imgs.new_tensor(imgs.shape[1:]).prod().float().sqrt()
    return torch.max(std, std_min)


def normalize_zero_mean(imgs):
    mean = imgs.view(imgs.shape[0], -1).mean(dim=1).view(
        imgs.shape[0], 1, 1, 1)
    return F.normalize(imgs-mean, dim=[1, 2, 3], p=2)


def _cifar_meanstd_normalize(x):
    f = transforms.Normalize([0.5071, 0.4865, 0.4409],
                             [0.2673, 0.2564, 0.2762])
    x = f(x)
    return x


class DefenseJARN_AT(torch.nn.Module):
    meta_path = 'checkpoints/JARN_AT/model/checkpoint-159999.meta'
    tf_ckpt = 'checkpoints/JARN_AT/model/checkpoint-159999'
    pt_path = 'checkpoints/JARN_AT/model/model.pt'

    def __init__(self, load_tf=False):
        super(DefenseJARN_AT, self).__init__()
        self.base_model = WideResNet(depth=34, widen_factor=10, num_classes=10)
        if load_tf:
            self.load_tf_weights()
            self.base_model.eval()
            # pt_out = self.forward(torch.ones(1, 3, 32, 32))
            # print(pt_out)
            # diff = np.abs(self.tf_out-pt_out.detach().cpu().numpy())
            # print(diff, diff.mean())
        else:
            self.load_pt_weights()
        self.base_model.eval()

    def _pre_process(self, x):
        return normalize_zero_mean(x)

    def forward(self, x):
        x = self._pre_process(x)
        return self.base_model(x)

    def load_pt_weights(self):
        self.base_model.load_state_dict(torch.load(self.pt_path))

    def load_tf_weights(self):
        g = tf.Graph()

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.meta_path)
            saver.restore(sess, self.tf_ckpt)
            graph = tf.get_default_graph().as_graph_def()
            # all_vars = tf.trainable_variables()
            all_vars = tf.global_variables()
            tf_names = []

            all_names = [op.name for op in tf.get_default_graph().get_operations()]
            for v in all_names:
                if v.startswith('gradients'):
                    continue
                elif v.startswith('save'):
                    continue
                elif v.startswith('Momentum') or 'Momentum' in v:
                    continue
                print(v)
            for v in all_vars:
                print(v)
                tf_names.append(v.name)

            # Load Weights
            for name, param in self.base_model.named_parameters():
                tf_key = get_tf_key(name)
                for v in all_vars:
                    if v.name == tf_key:
                        w = sess.run(v)
                        w = torch.FloatTensor(w)
                        if 'fc' not in name and 'bn' not in name:
                            w = w.permute((3, 2, 0, 1))
                        elif 'fc.weight' in name:
                            w = w.permute((1, 0))
                        if param.shape != w.shape:
                            print(name, param.shape, 'tf_key', w.shape)
                        assert(w.shape == param.shape)
                        param.data = w
                        print('%s Loaded with %s' % (name, tf_key))

            # # Update BN Mean/Var
            modules = list(self.base_model.named_modules())
            for name, module in self.base_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    tfbn_mean, tfbn_var = get_tf_bn_mean_var(name)
                    mean_updated, var_updated = False, False
                    for v in all_vars:
                        if v.name == tfbn_mean:
                            bn_mean = sess.run(tfbn_mean)
                            # print(v.name, bn_mean)
                            # print(module.running_mean)
                            bn_mean = torch.from_numpy(bn_mean)
                            assert(module.running_mean.shape == bn_mean.shape)
                            module.running_mean.data.copy_(bn_mean)
                            mean_updated = True
                        if v.name == tfbn_var:
                            bn_var = sess.run(tfbn_var)
                            # print(v.name, bn_var)
                            bn_var = torch.from_numpy(bn_var)
                            assert(module.running_var.shape == bn_var.shape)
                            module.running_mean.data.copy_(bn_var)
                            var_updated = True
                    print('%s Mean/Var with %s %s' % (name, tfbn_mean, tfbn_var))
                    assert((mean_updated and var_updated) == True)

            # dg = tf.get_default_graph()
            # # is_train = dg.get_tensor_by_name('is_train:0')
            # x = dg.get_tensor_by_name('classifier/input/Placeholder:0')
            # l = dg.get_tensor_by_name('classifier/logit/xw_plus_b:0')
            # x_in = np.ones((1, 32, 32, 3))
            # out = sess.run([l], feed_dict={x: x_in})
            # self.tf_out = out[0]
            # print(out[0])
            # sess.close()
            torch.save(self.base_model.state_dict(), self.pt_path)
        return

    def save(self):
        torch.save(self.base_model.state_dict(), self.pt_path)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, l_idx=0, dropRate=0.0,):
        super(BasicBlock, self).__init__()
        self.pad = (out_planes - in_planes)//2
        self.bn1 = nn.BatchNorm2d(in_planes, eps=0.001)
        self.relu1 = nn.LeakyReLU(0.1)
        # https://stackoverflow.com/questions/61422046/resnet-model-of-pytorch
        # -and-tensorflow-give-different-results-when-stride-2
        if stride == 1:
            self.pad1 = None
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
        elif stride == 2:
            self.pad1 = nn.ZeroPad2d((0, 1, 0, 1))  # 0,1,0,1
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                   stride=stride, padding=0, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.l_idx = l_idx
        self.mode = 'residual_only_activation'
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.AvgPool2d(
            kernel_size=stride,
            stride=stride,
            padding=0) or None

    def forward(self, x):
        if self.mode == 'residual_only_activation':
            out = self.relu1(self.bn1(x))
        elif self.mode == 'shared_activation':
            x = self.relu1(self.bn1(x))
            out = x

        if self.pad1 is not None:
            # out = self.pad1(out)
            out = self.conv1(self.pad1(out))
        else:
            out = self.conv1(out)
        # out = self.relu2(self.bn2(out))
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.equalInOut:
            out = torch.add(x, out)
        else:
            x = self.convShortcut(x)
            padding_size = (0, 0, 0, 0, self.pad, self.pad)
            x = F.pad(input=x, pad=padding_size, mode='constant', value=0)
            out = torch.add(x, out)
        self.c = out
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
                               stride=1, padding=1, bias=False)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1,
                                   dropRate)
        self.block1.layer[0].mode = 'shared_activation'
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2,
                                   dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2,
                                   dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], eps=0.001)
        self.relu = nn.LeakyReLU(0.1)
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
        # TF version use train mode
        # self.train()
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

    def _check_out(self, out):
        c = out.detach().cpu().numpy()
        self.pt_out = c


if __name__ == '__main__':
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    from torch.utils.data import DataLoader
    test = DefenseJARN_AT(load_tf=True).to(device)
    data = CIFAR10(root='/data/projects/punim0784/datasets',
                   train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=data, pin_memory=True,
                              batch_size=128, drop_last=False,
                              num_workers=4,
                              shuffle=True)
    test_data = CIFAR10(root='/data/projects/punim0784/datasets',
                        train=False, download=True,
                        transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, pin_memory=True,
                             batch_size=128, drop_last=False,
                             num_workers=4,
                             shuffle=True)
    # for e in range(10):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        test.train()
        logits = test(images)

    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        test.eval()
        with torch.no_grad():
            logits = test(images)
        _, predictions = torch.max(logits, 1)
        correct += (predictions == labels).sum().item()
    print(correct)
    test.save()
