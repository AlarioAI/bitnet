# BITNET
![bitnet](/assets/main_image.png)
----
This repository not only provides PyTorch implementations for training and evaluating 1.58-bit neural networks but also includes a unique integration where the experiments conducted automatically update a LaTeX-generated paper. By applying 1.58-bit quantization to convolutional neural networks and building upon transformative research in the field, this project extends beyond simple implementation to create a living document that evolves with ongoing experimentation. Our approach broadens the application of concepts introduced in seminal papers, fostering both reproducibility and real-time documentation of findings.


## Papers
* **BitNet: Scaling 1-bit Transformers for Large Language Models:** This paper introduced the simple yet effective concept of replacing linear projections with 1-bit counterparts. We extend this methodology to convolutional layers to explore broader applications in neural network architectures.
    * [Read the paper](https://arxiv.org/pdf/2310.11453.pdf)
* **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits:** This paper extends the BitNet architecture to use 1.58-bit quantization and introduces novel quantization functions. Our work utilizes these innovations across different types of neural networks.
    * [Read the paper](https://arxiv.org/pdf/2402.17764.pdf)



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

### `BitConv2d`
```python
import torch
from bitnet.nn.bitconv2d import BitConv2d

_input = torch.randn(64, 3, 32, 32)
layer = BitConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
output = layer(_input)

print(output.shape)
```
----

## Examples:

### Run one experiment at a time, for example
1. **Feedforward** `MNIST`
WIP
```

### Run all experiments with different seeds and update the `tex` file:
```sh
python run_experiments.py
python generate_table_results.py
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
