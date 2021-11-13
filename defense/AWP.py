"""
Based on code from https://github.com/csdongxian/AWP
"""
from . import utils
from models.wideresnet import WideResNet
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class DefenseAWP(torch.nn.Module):
    file_id = '1sSjh4i2imdoprw_JcPj2cZzrJm0RIRI6'
    destination = 'checkpoints/AWP/'

    def __init__(self):
        super(DefenseAWP, self).__init__()
        # Download pretrained weights
        filename = os.path.join(self.destination, 'RST-AWP_linf_wrn28-10.pt')
        if not os.path.exists(filename):
            if not os.path.exists(self.destination):
                os.makedirs(self.destination, exist_ok=True)
            utils.download_file_from_google_drive(self.file_id, filename)
        checkpoint = torch.load(filename)
        state_dict = checkpoint.get('state_dict')

        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s
        state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}

        # Load Weights
        self.base_model = WideResNet(depth=28, widen_factor=10, num_classes=10)
        self.base_model = self.base_model.to(device)
        self.base_model.load_state_dict(state_dict, strict=False)
        self.base_model.eval()

    def forward(self, x):
        return self.base_model(x)
