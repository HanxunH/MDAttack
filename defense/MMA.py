"""
Based on code from https://github.com/BorealisAI/mma_training
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def batch_per_image_standardization(imgs):
    # replicate tf.image.per_image_standardization, but in batch
    assert imgs.ndimension() == 4
    mean = imgs.view(imgs.shape[0], -1).mean(dim=1).view(
        imgs.shape[0], 1, 1, 1)
    return (imgs - mean) / batch_adjusted_stddev(imgs)


def batch_adjusted_stddev(imgs):
    # for batch_per_image_standardization
    std = imgs.view(imgs.shape[0], -1).std(dim=1).view(imgs.shape[0], 1, 1, 1)
    std_min = 1. / imgs.new_tensor(imgs.shape[1:]).prod().float().sqrt()
    return torch.max(std, std_min)


class DefenseMMA(nn.Module):
    path = 'checkpoints/MMA/cifar10-Linf-MMA-32-sd0/model_best.pt'

    def __init__(self):
        super(DefenseMMA, self).__init__()
        checkpoint = torch.load(self.path)
        state_dict = checkpoint.get('model', checkpoint)

        def strip_data_parallel(s):
            if s.startswith('1.'):
                return s[len('1.'):]
            else:
                return s
        state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}

        # Load Weights
        self.base_model = WideResNet(28, 10, widen_factor=4)
        self.base_model = self.base_model.to(device)
        self.base_model.load_state_dict(state_dict)
        self.base_model.eval()

    def forward(self, x):
        x = batch_per_image_standardization(x)
        return self.base_model(x)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.drop_rate = drop_rate
        self.in_out_equal = (in_planes == out_planes)

        if not self.in_out_equal:
            self.conv_shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride,
                padding=0, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        if not self.in_out_equal:
            x = self.conv_shortcut(out)
        out = self.relu2(self.bn2(self.conv1(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out += x
        return out


class ConvGroup(nn.Module):
    def __init__(
            self, num_blocks, in_planes, out_planes, block, stride,
            drop_rate=0.0):
        super(ConvGroup, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, num_blocks, stride, drop_rate)

    def _make_layer(
            self, block, in_planes, out_planes, num_blocks, stride, drop_rate):
        layers = []
        for i in range(int(num_blocks)):
            layers.append(
                block(in_planes=in_planes if i == 0 else out_planes,
                      out_planes=out_planes,
                      stride=stride if i == 0 else 1,
                      drop_rate=drop_rate)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0,
                 color_channels=3, block=BasicBlock):
        super(WideResNet, self).__init__()
        num_channels = [
            16, int(16 * widen_factor),
            int(32 * widen_factor), int(64 * widen_factor)]
        assert((depth - 4) % 6 == 0)
        num_blocks = (depth - 4) / 6

        self.conv1 = nn.Conv2d(
            color_channels, num_channels[0], kernel_size=3, stride=1,
            padding=1, bias=False)
        self.convgroup1 = ConvGroup(
            num_blocks, num_channels[0], num_channels[1], block, 1, drop_rate)
        self.convgroup2 = ConvGroup(
            num_blocks, num_channels[1], num_channels[2], block, 2, drop_rate)
        self.convgroup3 = ConvGroup(
            num_blocks, num_channels[2], num_channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(num_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]
        self.latent = False

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                n = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
                mod.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(mod, nn.BatchNorm2d):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.Linear):
                mod.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.convgroup1(out)
        out = self.convgroup2(out)
        # Expose features in the final residual blocks
        sequential = self.convgroup3.layer
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
