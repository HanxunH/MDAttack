"""
Based on code from https://github.com/HanxunH/RobustWRN
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from . import utils
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes,
                          out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class RobustWideResNet(nn.Module):
    def __init__(self, num_classes=10, channel_configs=[16, 160, 320, 640],
                 depth_configs=[5, 5, 5], stride_config=[1, 2, 2],
                 drop_rate_config=[0.0, 0.0, 0.0]):
        super(RobustWideResNet, self).__init__()
        assert len(channel_configs) - \
            1 == len(depth_configs) == len(
                stride_config) == len(drop_rate_config)
        self.channel_configs = channel_configs
        self.depth_configs = depth_configs
        self.stride_config = stride_config
        self.latent = False
        self.stem_conv = nn.Conv2d(3, channel_configs[0], kernel_size=3,
                                   stride=1, padding=1, bias=False)
        self.blocks = nn.ModuleList([])
        for i, stride in enumerate(stride_config):
            self.blocks.append(NetworkBlock(block=BasicBlock,
                                            nb_layers=depth_configs[i],
                                            in_planes=channel_configs[i],
                                            out_planes=channel_configs[i+1],
                                            stride=stride,
                                            dropRate=drop_rate_config[i],))

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channel_configs[-1])
        self.relu = nn.ReLU(inplace=True)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel_configs[-1], num_classes)
        self.fc_size = channel_configs[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.stem_conv(x)
        out = self.blocks[0](out)
        out = self.blocks[1](out)
        sequential = self.blocks[2].layer
        features = []
        for s in sequential:
            out = s(out)
            features.append(out)
        out = self.relu(self.bn1(out))
        out = self.global_pooling(out)
        out = out.view(-1, self.fc_size)
        pooled = out
        out = self.fc(out)
        if self.latent:
            features += [pooled, out]
            return features
        return out


class DefenseRobustWRN(torch.nn.Module):
    file_id = '1A13xrwItjJTfxyK1kiuYWSgxq0yYYc7D'
    destination = 'checkpoints/RobustWRN/'

    def __init__(self):
        super(DefenseRobustWRN, self).__init__()
        # Download pretrained weights
        filename = os.path.join(self.destination, 'WRN-34-R-EMA.pt')
        if not os.path.exists(filename):
            if not os.path.exists(self.destination):
                os.makedirs(self.destination, exist_ok=True)
            utils.download_file_from_google_drive(self.file_id, filename)
        checkpoint = torch.load(filename)

        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s
        checkpoint = {strip_data_parallel(k): v for k, v in checkpoint.items()}

        # Load Weights
        self.base_model = RobustWideResNet(
            num_classes=10, channel_configs=[16, 320, 640, 512],
            depth_configs=[5, 5, 5], stride_config=[1, 2, 2])
        self.base_model = self.base_model.to(device)
        self.base_model.load_state_dict(checkpoint)
        self.base_model.eval()

    def forward(self, x):
        return self.base_model(x)
