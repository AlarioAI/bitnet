# BITNET
![bitnet](/assets/main_image.png)
PyTorch implementations for training and evaluating 1.58-bits neural networks. It covers methods and models from the following research papers:


## Papers
* [BitNet: Scaling 1-bit Transformers for Large Language Models:](https://arxiv.org/pdf/2310.11453.pdf)
    "The implementation of the BitNet architecture is quite simple, requiring only the replacement of linear projections (i.e., nn.Linear in PyTorch) in the Transformer. " -- BitNet is really easy to implement just swap out the linears with the BitLinear modules!
* [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/pdf/2402.17764.pdf)
    **Extends BitNet to 1.58 bits:** Introduces modifications to the original BitNet architecture, including a novel quantization function (absmean) for the weights and a revised approach for handling activation outputs. These changes enable training with 1.58-bit weights while maintaining efficiency and performance.


## Installation

### From source
```sh
git clone git@github.com:AlarioAI/bitnet.git
cd bitnet
python3.11 -m pip install -e .
```

### Direct (main)
```sh
python3.11 -m pip install https://github.com/AlarioAI/bitnet.git
```

## Usage:

### `BitLinear`
```python
import torch
from bitnet.nn.bitlinear import BitLinear

_input = torch.randn(10, 512)
layer = BitLinear(512, 400)
output = layer(_input)

print(output)
```
----

## Examples:

1. **Feedforward** `MNIST`
Train a one-hidden layer 1.58bits neural network on the MNIST dataset
```sh
python examples/mnist_ff_example.py
```
2. **LeNet5** `MNIST`
Train the classic LeNet5 with 1.58bits linear and convolutional layers
```sh
python examples/mnist_lenet5_example.py
```

# License
MIT

# Citations
```bibtex
@misc{2310.11453,
Author = {Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Huaijie Wang and Lingxiao Ma and Fan Yang and Ruiping Wang and Yi Wu and Furu Wei},
Title = {BitNet: Scaling 1-bit Transformers for Large Language Models},
Year = {2023},
Eprint = {arXiv:2310.11453},
}

@misc{2402.17764,
Author = {Shuming Ma Hongyu Wang Lingxiao Ma Lei Wang Wenhui Wang Shaohan Huang Li Dong Ruiping Wang Jilong Xue Furu Wei},
Title = {The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (or Maybe Not Quite)},
Year = {2024},
Eprint = {arXiv:2402.17764},
}
```
