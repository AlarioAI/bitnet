from typing import Callable

import torch.nn as nn
from torch import Tensor

from bitnet.nn.bitconv2d import BitConv2d
from bitnet.nn.bitlinear import BitLinear


class LeNet(nn.Module):
    def __init__(
            self,
            linear_layer: Callable,
            conv_layer: Callable,
            num_classes: int,
            in_channels: int,
            input_size: int,
        ) -> None:
        super(LeNet, self).__init__()

        self.linear_layer = linear_layer
        self.conv_layer = conv_layer
        match input_size:
            case 28:
                size_first_linlayer: int = 400
            case 32:
                size_first_linlayer = 576
            case 64:
                size_first_linlayer = 3136
            case 96:
                size_first_linlayer = 7744
            case _:
                raise ValueError(f"Unsupported input size: {input_size}")

        self.layer1 = nn.Sequential(
            self.conv_layer(in_channels, 6, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            self.conv_layer(6, 16, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc = self.linear_layer(size_first_linlayer, 120)
        self.relu = nn.ReLU()
        self.fc1 = self.linear_layer(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = self.linear_layer(84, num_classes)


    @property
    def __name__(self) -> str:
        if self.linear_layer == BitLinear or self.conv_layer == BitConv2d:
            return "BitNet"
        return "FloatNet"


    def forward(self, x) -> Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.flatten(start_dim=1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
