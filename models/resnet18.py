"""
ImageNet-style ResNet (standard 7x7-stride-2 stem + adaptive average pool), for inputs larger
than 32x32.

models/resnet.py's ResNet is CIFAR-style: forward() ends with a hardcoded avg_pool2d(out, 4),
which only collapses to a single feature vector when the input is 32x32 (conv1 keeps the
resolution, then layer2/3/4 each halve it: 16x16 -> 8x8 -> 4x4). tiny_split.py loads TinyImageNet
at 224x224, where that hardcoded pool breaks (layer4's output is 28x28, not 4x4). This file reuses
the same BasicBlock but swaps in the standard ImageNet stem and an adaptive average pool, which
collapses to (B, latent_dims) at any input resolution.
"""
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import BasicBlock


class ResNetImageNet(nn.Module):
    def __init__(self, block, num_blocks):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.latent_dims = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


def ResNet18():
    return ResNetImageNet(BasicBlock, [2, 2, 2, 2])
