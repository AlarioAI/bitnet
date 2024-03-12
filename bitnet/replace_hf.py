from torch import nn

from bitnet.bitlinear import BitLinear


def replace_linears_in_hf(
    model,
):
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
            replace_linears_in_hf(module)
