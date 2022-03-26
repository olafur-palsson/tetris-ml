from torch import nn, optim

from ai.config.neural_net_config import NetworkConfig, LayerConfig, SGDConfig
from ai.lib.network_manager import Network


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
            if layer_config.type == 'linear':
                last_length = layer_config.linear.nodes
        layers.append(nn.Linear(last_length, network_config.output))
        if not network_config.no_sigmoid:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    @staticmethod
    def _create_layer(
            input_length,
            layer_config: LayerConfig) -> nn.Module:
        if layer_config.type == 'relu':
            return nn.ReLU()
        elif layer_config.type == 'linear':
            return nn.Linear(
                input_length,
                layer_config.linear.nodes,
                bias=True)
        elif layer_config.type == 'conv':
            return nn.Conv2d(
                layer_config.conv.input_channels,
                layer_config.conv.output_channels,
                kernel_size=layer_config.conv.kernel_size,
                stride=layer_config.conv.stride)

