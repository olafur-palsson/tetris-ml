from typing import Union
from torch import nn, optim

from ai.config.neural_net_config import NetworkConfig, LayerConfig, SGDConfig
from ai.lib.network import Network


class NetworkBuilder:

    @classmethod
    def create(cls, network_config: NetworkConfig) -> Network:
        model = cls._create_layer_sequence(network_config)
        optimizer = cls._create_optimizer(model, network_config.sgd)
        prefix = '-'.join([cls.namespace, network_config.id, '.pt'])
        return Network(
            id=prefix,
            model=model,
            optimizer=optimizer)

    @staticmethod
    def _create_optimizer(model, sgd_config: SGDConfig):
        return optim.SGD(
            model.parameters(),
            momentum=sgd_config.momentum,
            lr=sgd_config.learning_rate)

    @staticmethod
    def _create_layer_sequence(
            cls,
            input_length: int,
            network_config: NetworkConfig) -> nn.Sequential:
        layers = []
        last_length = input_length
        for layer_config in network_config.hidden_layers:
            layer = cls._create_layer(last_length, layer_config)
            layers.append(layer)
            last_length = layer_config.nodes
        layers.append(nn.Linear(last_length, network_config.output))
        return nn.Sequential(*layers)

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
