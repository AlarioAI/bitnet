import pytest

import torch
import torch.nn as nn
from torch import Tensor
from bitnet.nn.bitlinear import BitLinear


@pytest.fixture
def bit_linear_setup():
    return BitLinear(10, 5, True, 8)


def test_initialization():
    in_features = 10
    out_features = 5
    num_bits = 8
    bit_linear = BitLinear(in_features, out_features, True, num_bits)

    assert bit_linear.in_features == in_features
    assert bit_linear.out_features == out_features
    assert bit_linear.quantization_range == 2 ** (num_bits - 1)
    assert bit_linear.weight is not None
    assert bit_linear.bias is not None


def test_weight_quantization():
    bit_linear = BitLinear(10, 5, num_bits=8)
    original_weights = torch.randn(5, 10)
    bit_linear.weight = torch.nn.Parameter(original_weights)
    weights_gamma: float = original_weights.abs().mean().item()

    quantized_weights = bit_linear.ste_weights(weights_gamma)

    unique_values = torch.unique(quantized_weights)
    assert all(val in [-1, 0, 1] for val in unique_values)
    assert quantized_weights.shape == original_weights.shape

    quantized_weights.backward(torch.ones_like(quantized_weights), retain_graph=True)
    assert bit_linear.weight.grad is not None
    assert not torch.allclose(bit_linear.weight.grad, torch.zeros_like(bit_linear.weight.grad))


def test_forward_shape(bit_linear_setup):
    input_tensor = torch.randn(1, 10)
    output = bit_linear_setup(input_tensor)
    assert output.shape == (1, 5)


def test_activation_quantization(bit_linear_setup):
    input_tensor = torch.randn(1, 10)
    normalized_input: Tensor = nn.functional.layer_norm(input_tensor, (input_tensor.shape[1:]))
    input_gamma: float = normalized_input.abs().max().item()

    quantized_activations = bit_linear_setup.quantize_activations(input_tensor, input_gamma)
    quantization_range = bit_linear_setup.quantization_range
    assert torch.all(quantized_activations <= quantization_range)
    assert torch.all(quantized_activations >= -quantization_range)


def test_activation_dequantization(bit_linear_setup):
    input_tensor = torch.randn(1, 10)
    normalized_input: Tensor = nn.functional.layer_norm(input_tensor, (input_tensor.shape[1:]))
    input_gamma: float = normalized_input.abs().max().item()
    beta: float = bit_linear_setup.weight.abs().mean().item()

    quantized_activations = bit_linear_setup.quantize_activations(input_tensor, input_gamma)
    dequantized_activations = bit_linear_setup.dequantize_activations(quantized_activations, input_gamma, beta)

    assert dequantized_activations.shape == input_tensor.shape


def test_forward_pass_accuracy(bit_linear_setup):
    input_tensor = torch.randn(1, 10)
    output = bit_linear_setup(input_tensor)
    assert output is not None
    assert output.shape == (1, bit_linear_setup.out_features)


def test_integration_with_sequential():
    model = torch.nn.Sequential(
        BitLinear(10, 5, True, 8),
        torch.nn.ReLU(),
        BitLinear(5, 2, True, 8)
    )
    input_tensor = torch.randn(1, 10)
    output = model(input_tensor)
    assert output.shape == (1, 2)

    target = torch.randn_like(output)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    assert loss is not None


if __name__ == "__main__":
    pytest.main()
