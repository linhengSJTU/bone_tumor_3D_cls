import os
import logging
import torch
import torch.nn as nn
from .ClsModel.resnet import *

pretrain_path = ''

class ClsNet(nn.Module):
    def __init__(self, arch, resnet_shortcut, num_classes):
        super(ClsNet, self).__init__()
        self.arch = arch
        self.model = eval(arch)(shortcut_type=resnet_shortcut, num_classes=num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

    def load_pretrained_weights(self):
        arch = self.arch
        logging.info('Loading pretrained weights...')
        pretrain = torch.load(os.path.join(pretrain_path, arch + '_23dataset.pth'))
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrain['state_dict'].items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        logging.info('Finished loading pretrained weights!')


def build_model(config):
    arch = config['arch']
    num_classes = len(config['Data_CLASSES']) - 1
    resnet_shortcut = config['resnet_shortcut']
    return ClsNet(arch, resnet_shortcut, num_classes)