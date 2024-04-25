from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from bitnet.nn.bitlinear import BitLinear


class Feedforward(nn.Module):
    def __init__(
            self,
            linear_layer: Callable,
            input_size: int,
            hidden_size: int,
            num_classes: int
        ) -> None:
        super(Feedforward, self).__init__()
        self.linear_layer = linear_layer
        self.fc1 = linear_layer(input_size, hidden_size)
        self.fc2 = linear_layer(hidden_size, num_classes)


    @property
    def __name__(self) -> str:
        return "BitNet" if self.linear_layer == BitLinear else "FloatNet"


    def forward(self, _input: Tensor) -> Tensor:
        out = _input.flatten(start_dim=1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out
