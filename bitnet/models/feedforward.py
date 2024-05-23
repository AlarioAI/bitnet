import torch
import torch.nn as nn
from torch import Tensor


class Feedforward(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_classes: int
        ) -> None:
        super(Feedforward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, _input: Tensor) -> Tensor:
        out = _input.flatten(start_dim=1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


