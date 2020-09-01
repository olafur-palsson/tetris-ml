import os
from typing import Dict

import torch

from ai.config.neural_net_config import Config, NetworkConfig
from ai.lib.network import Network
from ai.lib.network_builder import NetworkBuilder


EXPORTS_FOLDER = './ai/exports/'


class NetworkManager(Network):
    name: str
    _config: Config
    _networks: Dict[str, Network]

    def __init__(self, config: Config):
        self._config = config
        try:
            self._networks = self._load()
        except:
            should_create = input('Unable to load, create? ')\
                .lower()\
                .startswith('y')
            if should_create:
                self._networks = self._create()
            else:
                raise Exception('Unable to load')

    def get(self, network_id: str):
        return self._networks[network_id]

    def zero_grad(self):
        for network in self._networks.values():
            network.optimizer.zero_grad()

    def step(self):
        for network in self._networks.values():
            network.optimizer.step()

    def save(self):
        self._touch_folder('export')
        for _, network in self._networks.items():
            self._touch_folder(network.id)
            print(self._create_model_name(network.id))
            torch.save(
                network.model,
                self._create_model_name(network.id))
            torch.save(
                network.optimizer.state_dict(),
                self._create_optimizer_name(network.id))

    def _load(self) -> Dict[str, Network]:
        networks = {}
        for _, network_config in self._config.networks.items():
            model = torch.load(self._create_model_name(network_config.id))
            optimizer = torch.optim.SGD(
                model.parameters(),
                momentum=network_config.sgd.momentum,
                lr=network_config.sgd.learning_rate)
            networks[network_config.id] = Network(
                optimizer=optimizer,
                model=model,
                id=network_config.id)
        return networks

    def _create(self):
        print(self._config.networks)
        return {
            network.id: NetworkBuilder.create(network)
            for _, network in self._config.networks.items()
        }

    @property
    def _folder_path(self):
        return os.path.join(EXPORTS_FOLDER, self._config.name)

    def _create_optimizer_name(self, network_id: str):
        return f'{os.path.join(self._folder_path, network_id)}_optimizer.pt'

    def _create_model_name(self, network_id: str):
        return f'{os.path.join(self._folder_path, network_id)}_model.pt'

    def _touch_folder(self, network_id: str):
        try:
            os.makedirs(self._folder_path)
        except:
            pass
