'''
Based on code from https://github.com/ShawnXYang/AT_HE/tree/master/CIFAR-10
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
    f = tf.Normalize([0.4914, 0.4822, 0.4465],
                     [0.2471, 0.2435, 0.2616])
    x = f(x)
    return x


class DefenseATHE(torch.nn.Module):
    file_id = 'http://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/weights/model-wideres-pgdHE-wide10.pt'
    destination = 'checkpoints/ATHE/'

    def __init__(self):
        super(DefenseATHE, self).__init__()
        # Download pretrained weights
        filename = os.path.join(self.destination, 'wideres-pgdHE-wide10.pt')
        if not os.path.exists(filename):
            if not os.path.exists(self.destination):
                os.makedirs(self.destination, exist_ok=True)
            utils.download_file_from_url(self.file_id, filename)
        checkpoint = torch.load(filename)
        state_dict = checkpoint.get('state_dict')

        def strip_data_parallel(s):
            if s.startswith('module'):
                return s[len('module.'):]
            else:
                return s
        state_dict = {strip_data_parallel(k): v for k, v in state_dict.items()}

        # Load Weights
        self.base_model = WideResNet(depth=34, widen_factor=10, num_classes=10)
        self.base_model = self.base_model.to(device)
        self.base_model.load_state_dict(state_dict, strict=False)
        self.base_model.eval()

    def forward(self, x):
        x = _cifar_meanstd_normalize(x)
        return self.base_model(x)
