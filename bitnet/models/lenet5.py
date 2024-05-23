import torch.nn as nn
from torch import Tensor


class LeNet(nn.Module):
    def __init__(
            self,
            num_classes: int,
            in_channels: int,
            input_size: int,
        ) -> None:
        super(LeNet, self).__init__()

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
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(size_first_linlayer, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)


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
