"""
Based on code from https://github.com/Adv-Interp/adv_interp
"""

import math
import torch
import torch.nn as nn
import os
import torchvision.transforms as tf
from . import utils
from models.wideresnet import WideResNet

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def _cifar_meanstd_normalize(x):
    f = tf.Normalize([0.5, 0.5, 0.5],
                     [0.5, 0.5, 0.5])
    x = f(x)
    return x


class DefenseADVInterp(torch.nn.Module):
    file_id = '1NWYmLAArzstzaknO1L0ZMni0_8UaSpTo'
    destination = 'checkpoints/ADVInterp/'

    def __init__(self):
        super(DefenseADVInterp, self).__init__()
        filename = os.path.join(self.destination, 'cifar10.pt')
        if not os.path.exists(filename):
            if not os.path.exists(self.destination):
                os.makedirs(self.destination, exist_ok=True)
            utils.download_file_from_google_drive(self.file_id, filename)
        checkpoint = torch.load(filename)
        state_dict = checkpoint.get('net', checkpoint)
        self.base_model = WideResNet(depth=28, widen_factor=10, num_classes=10)

        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s
        state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
        self.base_model.load_state_dict(state_dict)
        self.base_model.eval()

    def forward(self, x):
        x = _cifar_meanstd_normalize(x)
        return self.base_model(x)
