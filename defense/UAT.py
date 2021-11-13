"""
Based on code from https://github.com/deepmind/deepmind-research
"""

import math
import parse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow.compat.v1 as tf
import numpy as np
import torchvision.transforms as transforms
import tensorflow_hub as hub
from tensorflow.python.tools import inspect_checkpoint

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def _cifar_meanstd_normalize(x):
    cifar_means = torch.tensor([125.3/255, 123.0/255, 113.9/255])
    cifar_devs = torch.tensor([63.0/255, 62.1/255, 66.7/255])
    f = transforms.Normalize([125.3/255, 123.0/255, 113.9/255],
                             [63.0/255, 62.1/255, 66.7/255])
    x = f(x)
    return x


def get_tf_key(torch_name):
    if torch_name == 'conv1.weight':
        return "wide_res_net/init_conv/w"
    elif torch_name.startswith('block') and 'bn' in torch_name:
        f = "block{:d}.layer.{:d}.bn{:d}.{}"
        parsed = parse.parse(f, torch_name)
        b = parsed[0] - 1
        l = parsed[1]
        n = parsed[2]
        t = parsed[3]
        if t == 'weight':
            t = 'gamma'
        elif t == 'bias':
            t = 'beta'
        if n == 1:
            return "wide_res_net/resnet_lay_{:d}_block_{:d}/CrossReplicaBN/{}".format(b, l, t)
        elif n == 2:
            return "wide_res_net/resnet_lay_{:d}_block_{:d}/CrossReplicaBN_1/{}".format(b, l, t)
    elif '.convShortcut.weight' in torch_name:
        f = "block{:d}.layer.{:d}.convShortcut.weight"
        parsed = parse.parse(f, torch_name)
        b = parsed[0] - 1
        l = parsed[1]
        return 'wide_res_net/resnet_lay_{:d}_block_{:d}/shortcut_x/w'.format(b, l)
    elif torch_name.startswith('block') and 'conv' in torch_name:
        # print(torch_name)
        f = "block{:d}.layer.{:d}.conv{:d}.weight"
        parsed = parse.parse(f, torch_name)
        b = parsed[0] - 1
        l = parsed[1]
        n = parsed[2] - 1
        return 'wide_res_net/resnet_lay_{:d}_block_{:d}/conv_{:d}/w'.format(b, l, n)
    elif torch_name == 'bn1.weight':
        return 'wide_res_net/CrossReplicaBN/gamma'
    elif torch_name == 'bn1.bias':
        return 'wide_res_net/CrossReplicaBN/beta'
    elif torch_name == 'fc.weight':
        return "wide_res_net/linear/w"
    elif torch_name == 'fc.bias':
        return "wide_res_net/linear/b"


def get_tf_bn_mean_var(torch_name):
    m, b = None, None
    if torch_name.startswith('block') and 'bn' in torch_name:
        f = "block{:d}.layer.{:d}.bn{:d}"
        parsed = parse.parse(f, torch_name)
        b = parsed[0] - 1
        l = parsed[1]
        n = parsed[2]
        if n == 1:
            m = "wide_res_net/resnet_lay_{:d}_block_{:d}/CrossReplicaBN/accumulated_mean".format(
                b, l)
            v = "wide_res_net/resnet_lay_{:d}_block_{:d}/CrossReplicaBN/accumulated_var".format(
                b, l)
        elif n == 2:
            m = "wide_res_net/resnet_lay_{:d}_block_{:d}/CrossReplicaBN_1/accumulated_mean".format(
                b, l)
            v = "wide_res_net/resnet_lay_{:d}_block_{:d}/CrossReplicaBN_1/accumulated_var".format(
                b, l)
    elif torch_name == 'bn1':
        m = 'wide_res_net/CrossReplicaBN/accumulated_mean'
        v = 'wide_res_net/CrossReplicaBN/accumulated_var'
    return m, v


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def


class DefenseUAT(torch.nn.Module):
    tf_ckpt = 'checkpoints/UAT/unsupervised-adversarial-training_cifar10_wrn_106_1/frozen.pb'
    tf_path = 'checkpoints/UAT/unsupervised-adversarial-training_cifar10_wrn_106_1/'
    pt_path = 'checkpoints/UAT/UAT.pt'

    def __init__(self, load_tf=False):
        super(DefenseUAT, self).__init__()
        self.base_model = WideResNet(depth=106, widen_factor=8, num_classes=10)
        if load_tf:
            self.load_tf_weights()
            # self.base_model.eval()
            # logits = self.base_model(torch.ones((1,3,32,32)))
            # self.pt_out = self.base_model.pt_out
            # print('='*20)
            # print(self.tf_out)
            # print('='*20)
            # print(self.pt_out)
            # print('='*20)
            # diff = np.abs(self.tf_out - self.pt_out)
            # print(diff, diff.mean())
        else:
            self.load_pt_weights()
        self.base_model.eval()

    def load_tf_weights(self):
        with tf.Graph().as_default() as g:
            with tf.Session() as sess:
                with tf.gfile.GFile(self.tf_ckpt, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    sess.graph.as_default()
                graph_nodes = [n for n in graph_def.node]
                nodes = {}
                for t in graph_nodes:
                    print(t.name)
                    if t.op == 'Const':
                        nodes[t.name] = t
                        w = tensor_util.MakeNdarray(t.attr['value'].tensor)
                        print(w.shape, t.name)
                # print(sess.run('wide_res_net/resnet_lay_2_block_3/conv_0/w'))

                # Load Weights
                for name, param in self.base_model.named_parameters():
                    tf_key = get_tf_key(name)
                    if tf_key not in nodes.keys():
                        print(tf_key)
                    t = nodes[tf_key]
                    w = tensor_util.MakeNdarray(t.attr['value'].tensor)
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

                # Update BN Mean/Var
                modules = list(self.base_model.named_modules())
                for name, module in self.base_model.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        tfbn_mean, tfbn_var = get_tf_bn_mean_var(name)
                        if tfbn_mean not in nodes.keys():
                            print(tfbn_mean)
                        if tfbn_var not in nodes.keys():
                            print(tfbn_var)
                        t = nodes[tfbn_mean]
                        bn_mean = tensor_util.MakeNdarray(
                            t.attr['value'].tensor)
                        t = nodes[tfbn_var]
                        bn_var = tensor_util.MakeNdarray(
                            t.attr['value'].tensor)
                        bn_mean = torch.FloatTensor(bn_mean)
                        bn_var = torch.FloatTensor(bn_var)
                        module.running_mean = bn_mean
                        module.running_var = bn_var
                        print('%s Mean/Var with %s %s' %
                              (name, tfbn_mean, tfbn_var))

        # with tf.Session(graph=tf.Graph()) as sess:
        #     tf.saved_model.loader.load(sess, [], self.tf_path)
        #     dg = tf.get_default_graph()
        #     all_names = [op.name for op in tf.get_default_graph().get_operations()]
        #     for v in all_names:
        #         if v.startswith('gradients'):
        #             continue
        #         elif v.startswith('save'):
        #             continue
        #         elif v.startswith('Momentum') or 'Momentum' in v:
        #             continue
        #         print(v)
        #
        #     x = dg.get_tensor_by_name('Placeholder_2:0')
        #     check_name = 'wide_res_net/linear/add:0'
        #     l = dg.get_tensor_by_name(check_name)
        #     x_in = np.ones((1, 32, 32, 3))
        #     out = sess.run([l], feed_dict={x: x_in})
        #     self.tf_out = out[0]

        torch.save(self.base_model.state_dict(), self.pt_path)

    def load_pt_weights(self):
        self.base_model.load_state_dict(torch.load(self.pt_path))

    def forward(self, x):
        x = _cifar_meanstd_normalize(x)
        return self.base_model(x)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, l_idx=0, dropRate=0.0,):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.pad = (out_planes - in_planes)//2
        self.bn1 = nn.BatchNorm2d(in_planes, eps=0.00001)
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

        self.bn2 = nn.BatchNorm2d(out_planes, eps=0.00001)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.l_idx = l_idx
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride,
            padding=0, bias=False)

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.pad1 is not None:
            out = self.relu2(
                self.bn2(self.conv1(self.pad1(out if self.equalInOut else x))))
        else:
            out = self.relu2(
                self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.conv2(out)
        out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
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
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2,
                                   dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2,
                                   dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], eps=0.00001)
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
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        pooled = out
        out = self.fc(out)
        if self.latent:
            features += [pooled, out]
            return features
        return out


if __name__ == '__main__':
    test = DefenseUAT(load_tf=True)
