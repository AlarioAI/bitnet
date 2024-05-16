from torch import nn

from bitnet.nn.bitconv2d import BitConv2d
from bitnet.nn.bitlinear import BitLinear


def replace_linear_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(
                model,
                name,
                BitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                ),
            )
        else:
            replace_linear_layers(module)


def replace_conv2d_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(
                model,
                name,
                BitConv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=module.bias is not None,
                ),
            )
        else:
            replace_conv2d_layers(module)


def replace_layers(model):
    replace_linear_layers(model)
    replace_conv2d_layers(model)
