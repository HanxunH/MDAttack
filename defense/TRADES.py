"""
Based on code from https://github.com/yaodongyu/TRADES
"""

import torch
import os
from . import utils
from models.wideresnet import WideResNet

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class DefenseTRADES(torch.nn.Module):
    file_id = '10sHvaXhTNZGz618QmD5gSOAjO3rMzV33'
    destination = 'checkpoints/TRADES/'

    def __init__(self):
        super(DefenseTRADES, self).__init__()
        # Download pretrained weights
        filename = os.path.join(self.destination, 'model_cifar_wrn.pt')
        if not os.path.exists(filename):
            if not os.path.exists(self.destination):
                os.makedirs(self.destination, exist_ok=True)
            utils.download_file_from_google_drive(self.file_id, filename)
        checkpoint = torch.load(filename)

        # Load Weights
        self.base_model = WideResNet(depth=34, widen_factor=10, num_classes=10)
        self.base_model = self.base_model.to(device)
        self.base_model.load_state_dict(checkpoint, strict=False)

    def forward(self, x):
        return self.base_model(x)
