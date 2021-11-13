'''
Based on code from https://github.com/locuslab/robust_overfitting
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.transforms as tf
from models.wideresnet import WideResNet
from . import utils

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def _cifar_meanstd_normalize(x):
    f = tf.Normalize([0.5071, 0.4865, 0.4409],
                     [0.2673, 0.2564, 0.2762])
    x = f(x)
    return x


class DefenseOverfitting(torch.nn.Module):
    file_id = '12jPK-KXQc7-AF3w1zCbQxhBQAxm4KHzZ'
    destination = 'checkpoints/Overfitting/'

    def __init__(self):
        super(DefenseOverfitting, self).__init__()
        # Download pretrained weights
        filename = os.path.join(
            self.destination, 'cifar10_wide10_linf_eps8.pth')
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
        self.base_model = WideResNet(depth=34, widen_factor=10, num_classes=10)
        self.base_model = self.base_model.to(device)
        self.base_model.load_state_dict(checkpoint, strict=False)
        self.base_model.eval()

    def forward(self, x):
        x = _cifar_meanstd_normalize(x)
        return self.base_model(x)
