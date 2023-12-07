#*******************************************************************************#
#                                                                               #
#                                                                               #
#                             This code is unfinished.                          #
#                                                                               #
#                                                                               #
#*******************************************************************************#








import os
import sys
import time
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms



def conv3x3(in_channels, out_channels, stride):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        out = self.relu2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block_type, num_layers, num_classes=101):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block_type, 16, num_layers[0])
        self.layer2 = self._make_layer(block_type, 32, num_layers[1], 2)
        self.layer3 = self._make_layer(block_type, 64, num_layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block_type, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                    conv3x3(self.in_channels, out_channels, stride=stride),
                    nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        



if __name__ == "__main__":
    epochs = [30, 30, 20]
    batch_size = 128
    lr = 5e-4

    device = torch.device('cuda')

    data_root = "/scratch/ym2380/data/caltech101/"
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    dataset = torchvision.datasets.Caltech101(root=data_root, transform=transform, download=True)
    training_set, test_set = torch.utils.data.random_split(dataset, [8000, 1144])
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    network = ResNet(ResidualBlock, [2, 2, 2])
    network.to(device)
