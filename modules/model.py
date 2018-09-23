# coding=utf-8
# date: 2018-9-15,20:51:59
# name: smz

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable

from layers import LRN


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layers = nn.Sequential(OrderedDict([   # 107 x 107
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),     # 51 x 51
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),       # 25 x 25
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),   # 11 x 11
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),       # 5 x 5
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),  # 3 x 3
                                    nn.ReLU())),
            ('fc4', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear()))
        ]))

    def forward(self, input):
        pass

    def gather_params(self):
        pass

    def restore_params(self):
        pass
