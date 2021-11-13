"""
Based on code from https://github.com/hendrycks/pre-training
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.wideresnet import WideResNet
from . import utils

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def pre_process(x):
    x = x * 2 - 1
    return x


class DefensePreTrain(torch.nn.Module):
    file_id = '18jsSVQax2OH6rorx8L-iNaTWRV566a4S'
    destination = 'checkpoints/PreTrain/'

    def __init__(self):
        super(DefensePreTrain, self).__init__()
        # Download pretrained weights
        filename = os.path.join(
            self.destination, 'cifar10wrn_baseline_epoch_4.pt')
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
        self.base_model = WideResNet(depth=28, widen_factor=10, num_classes=10)
        self.base_model = self.base_model.to(device)
        self.base_model.load_state_dict(checkpoint)
        self.base_model.eval()

    def forward(self, x):
        x = pre_process(x)
        return self.base_model(x)
