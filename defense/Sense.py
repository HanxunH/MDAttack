"""
Based on code from https://openreview.net/forum?id=rJlf_RVKwr
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


class DefenseSense(torch.nn.Module):
    ckpt_path = 'checkpoints/Sense/SENSE_checkpoint300.dict'
    frozen_path = 'checkpoints/Sense/frozen.pt'

    def __init__(self):
        super(DefenseSense, self).__init__()
        checkpoint = torch.load(self.ckpt_path)
        state_dict = checkpoint.get('state_dict', checkpoint)
        self.base_model = WideResNet(depth=34, widen_factor=10, num_classes=10)

        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s
        state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}
        self.base_model.load_state_dict(state_dict, strict=False)
        self.base_model.eval()

    def forward(self, x):
        return self.base_model(x)

    def forward(self, x):
        return self.base_model(x)
