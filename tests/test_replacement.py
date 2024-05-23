from bitnet.layer_swap import replace_layers
from bitnet.models.feedforward import Feedforward
from bitnet.models.lenet5 import LeNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def test_linear_replacement():
    input_size: int = 784
    hidden_size: int = 256
    num_classes: int = 10

    original_model = Feedforward(input_size, hidden_size, num_classes)
    original_params = count_parameters(original_model)
    print(f"Original model parameters: {original_params}")

    replace_layers(original_model)
    replaced_params = count_parameters(original_model)
    print(f"Replaced model parameters: {replaced_params}")

    assert original_params == replaced_params, "Parameter count mismatch after replacing layers!"


def test_linear_and_conv_replacement():
    input_size:int  = 32
    num_classes:int = 10
    in_channels: int = 3
    original_model = LeNet(num_classes, in_channels=in_channels, input_size=input_size)
    original_params = count_parameters(original_model)
    print(f"Original model parameters: {original_params}")

    replace_layers(original_model)
    replaced_params = count_parameters(original_model)
    print(f"Replaced model parameters: {replaced_params}")

    assert original_params == replaced_params, "Parameter count mismatch after replacing layers!"
