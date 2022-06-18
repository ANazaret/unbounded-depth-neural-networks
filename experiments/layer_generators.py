from torch import nn
import torch.nn.functional as F


def generator_layers_cifar(layer_id: int, is_residual: bool = True) -> nn.Module:
    """Generator for the hidden layers of the UDN for CIFAR-10.

    At every call, generate the full network layer by layer until reaching the target `layer_id`, to get
    the correct dimensions.

    Parameters
    ----------
    layer_id: int
        Hidden layer to generate.
    is_residual: bool, default True
        Whether to have skip connection (residual net)

    Returns
    -------
    nn.Module:
        hidden layer of depth `layer_id` in the CNN for CIFAR classification

    """
    if layer_id == -1:
        return None, 3, 32

    FIRST_LAYER_CHANNEL = 64
    out_dim = 32
    if layer_id == 0:
        bloc = [
            nn.Conv2d(3, FIRST_LAYER_CHANNEL, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(FIRST_LAYER_CHANNEL),
            nn.ReLU(),
        ]
        return nn.Sequential(*bloc), FIRST_LAYER_CHANNEL, out_dim

    in_channel = FIRST_LAYER_CHANNEL
    total = 0
    for block_id in range(1, 4):
        n_of_blocks = [0, 3, 5, 1000][block_id]
        channels_of_block = 2 ** (block_id + 5)

        for i in range(n_of_blocks):
            total += 1
            if i == 0:
                stride = 2
            else:
                stride = 1

            out_channel = channels_of_block * BottleneckCNN.expansion
            out_dim //= stride

            if total == layer_id:
                bloc = BottleneckCNN(in_channel, channels_of_block, stride, is_residual)
                return bloc, out_channel, out_dim

            in_channel = out_channel


def generator_output_cifar(layer_id: int, generator_hidden_layers):
    """Generator for the output layers of the UDN for CIFAR-10.

    At every call, generate the hidden layer to obtain the correct input dimension.

    Parameters
    ----------
    layer_id: int
        Hidden layer to generate.
    generator_hidden_layers: callable
       Generator of the hidden layers.

    Returns
    -------
    nn.Module:
        output layer of depth L in the CNN for CIFAR classification

    """
    _, last_channels, last_dim = generator_hidden_layers(layer_id)
    last_hidden_size = (last_dim // 4) ** 2 * last_channels
    layers = [
        nn.AvgPool2d(4),
        nn.Flatten(),
        nn.Linear(last_hidden_size, 10),
    ]

    return nn.Sequential(*layers)


class BottleneckCNN(nn.Module):
    """Construction block for the CNN"""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, residual=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.is_residual = residual
        self.shortcut = nn.Sequential()
        if (stride != 1 or in_channels != self.expansion * out_channels) and self.is_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.is_residual:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


def make_generators_fcn(hidden_size, input_size, output_size):
    """Returns generators for a simple infinitely deep fully connected neural network of constant hidden dimension."""

    def hidden_layers(layer_id):
        if layer_id == -1:
            return None, input_size

        if layer_id == 0:
            return (
                nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU()),
                hidden_size,
            )

        return (
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()),
            hidden_size,
        )

    def output_layers(layer_id, hidden_layers):
        _, s = hidden_layers(layer_id)
        return nn.Linear(s, output_size)

    return hidden_layers, output_layers


def make_generators_fcn_DUN(hidden_size, input_size, output_size):
    """Returns generators for a simple infinitely deep fully connected neural network of constant hidden dimension.
    DUN is a bit delicate because the output layers are shared. The same layer needs to be returned from multiple calls
    of output_layers.
    """

    def main_layers(layer_id):
        if layer_id == -1:
            return None, input_size

        if layer_id == 0:
            return (
                nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU()),
                hidden_size,
            )

        return (
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()),
            hidden_size,
        )

    o_layer = nn.Linear(input_size, output_size)
    final_layer = nn.Linear(hidden_size, output_size)

    def output_layers(layer_id, hidden_layers):
        if layer_id >= 0:
            return final_layer
        else:
            return o_layer

    return main_layers, output_layers
