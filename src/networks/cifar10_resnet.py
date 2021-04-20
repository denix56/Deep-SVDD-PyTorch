import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from base.base_net import BaseNet


class CIFAR10_ResNet(BaseNet):

    def __init__(self):
        super().__init__()



        self.net = resnet18(pretrained=True)
        self.rep_dim = self.net.fc.in_features
        self.net = nn.Sequential(*list(self.net.children())[:-2])
        #self.net.fc = nn.Linear(self.net.fc.in_features, self.rep_dim, bias=False)


        # self.pool = nn.MaxPool2d(2, 2)
        #
        # self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        # self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        # self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        # self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        # self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        # self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        # self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        return self.net(x)