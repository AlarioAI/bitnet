import torch.nn as nn
from torch import Tensor

from bitnet.nn.bitlinear import BitLinear
from bitnet.nn.bitconv2d import BitConv2d


class LeNet(nn.Module):
    def __init__(
            self,
            linear_layer: callable,
            conv_layer: callable,
            num_classes: int,
            in_channels: int,
            input_size: int,
        ) -> None:
        super(LeNet, self).__init__()

        assert input_size in (28, 32)
        self.linear_layer = linear_layer
        self.conv_layer = conv_layer
        size_first_linlayer: int = 400 if input_size == 28 else 576

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
