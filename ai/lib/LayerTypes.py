from typing import Union
from torch import nn, optim

from ai.config.neural_net_config import NetworkConfig, LayerConfig, SGDConfig
from ai.lib.network_manager import Network


class Layers:
    LINEAR = nn.Linear
    RELU = nn.ReLU
    CONV2D = nn.Conv2d


class NetworkBuilder:

    @classmethod
    def create(cls, network_config: NetworkConfig) -> Network:
        model = cls._create_layer_sequence(network_config)
        optimizer = cls._create_optimizer(model, network_config.sgd)
        return Network(
            id=network_config.id,
            model=model,
            optimizer=optimizer)

    @staticmethod
    def _create_optimizer(model, sgd_config: SGDConfig):
        return optim.SGD(
            model.parameters(),
            momentum=sgd_config.momentum,
            lr=sgd_config.learning_rate)

    @classmethod
    def _create_layer_sequence(
            cls,
            network_config: NetworkConfig) -> nn.Sequential:
        layers = []
        last_length = network_config.input
        for layer_config in network_config.hidden_layers:
            layer = cls._create_layer(last_length, layer_config)
            layers.append(layer)
            last_length = layer_config.nodes
        layers.append(nn.Linear(last_length, network_config.output))
        return nn.Sequential(*layers)

    def _create_conv_layer(
            self,
            layer_config: LayerConfig) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(layer_config.input_channels, input_channels, kernel_size=layer_config.kernel_size, stride=layer_config.stride),
        )

    @staticmethod
    def _create_layer(
            input_length,
            layer_config: LayerConfig) -> Union[nn.ReLU, nn.Linear]:
        if layer_config.relu:
            return nn.ReLU()
        else:
            return nn.Linear(
                input_length,
                layer_config.nodes,
                bias=layer_config.bias)
