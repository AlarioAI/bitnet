import torch
import torch.nn as nn


class BitConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        num_bits: int = 8,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.eps:float = 1e-5
        self.quantization_range: int = 2 ** (num_bits - 1) # Q_b in the paper


    def ste_weights(self, weights_gamma: float) -> torch.Tensor:
        eps: float = 1e-7
        scaled_weights:torch.Tensor = self.weight / (weights_gamma + eps)
        binarized_input_no_grad: torch.Tensor = torch.clamp(torch.round(scaled_weights), min=-1, max=1)
        binarized_input_with_grad: torch.Tensor = (binarized_input_no_grad - self.weight).detach() + self.weight
        return binarized_input_with_grad


    def binarize_weights(self, weights_gamma: float) -> torch.Tensor:
        binarized_weights = self.ste_weights(weights_gamma)
        return binarized_weights


    def quantize_activations(self, _input:torch.Tensor, input_gamma: float) -> torch.Tensor:
        # Equation 4 BitNet paper
        quantized_input = torch.clamp(
                _input * self.quantization_range / input_gamma,
                -self.quantization_range + self.eps,
                self.quantization_range - self.eps,
            )
        return quantized_input


    def dequantize_activations(self, _input: torch.Tensor, input_gamma: float, beta: float) -> torch.Tensor:
        return _input * input_gamma * beta / self.quantization_range


    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        normalized_input: torch.Tensor = nn.functional.layer_norm(_input, (_input.shape[1:]))
        input_gamma: float = normalized_input.abs().max().item()
        weight_abs_mean: float = self.weight.abs().mean().item()

        binarized_weights = self.binarize_weights(weight_abs_mean)
        input_quant = self.quantize_activations(normalized_input, input_gamma)
        output = torch.nn.functional.conv2d(input_quant, binarized_weights, self.bias)
        output = self.dequantize_activations(output, input_gamma, weight_abs_mean)

        return output


def main():
    test_tensor = torch.randn(1, 3, 28, 28)
    layer = BitConv2d(3, 16, 3)
    out = layer(test_tensor)
    print(out)

if __name__ == "__main__":
    main()