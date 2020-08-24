import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from ai.config.neural_net_config import Config
from ai.lib.network_builder import NetworkBuilder


EXPORTS_FOLDER = './ai/exports/'


@dataclass
class Network:
    optimizer: any
    model: any
    id: any


class NetworkManager(Network):
    name: str
    _config: Config
    _networks: Dict[Network]

    def __init__(self, config: Config):
        self._config = config
        try:
            self._load()
        except:
            should_create = input('Unable to load, create?').lower() == 'y'
            if should_create:
                self._create()
            else:
                raise Exception('Unable to load')

    def get(self, network_id: str):
        return self._networks[network_id]

    def save(self):
        self.touch_folder('export')
        for _, network in self._networks:
            self.touch_folder(network.id)
            torch.save(
                network.model,
                self._create_model_name(network.id))
            torch.save(
                network.optimizer.state_dict(),
                self._create_optimizer_name(network.id))

    def _load(self):
        for _, network_config in self._config.networks:
            self.model = torch.load(self._create_model_name(network_config.id))
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                momentum=network_config.sgd.momentum,
                lr=network_config.sgd.learning_rate)

    def _create(self):
        return {
            network.id: NetworkBuilder.create(network)
            for _, network in self._config.networks
        }

    @property
    def _folder_path(self):
        return os.path.join(EXPORTS_FOLDER, self._config.name)

    def _create_optimizer_name(self, network: Network):
        return f'{os.path.join(self._folder_path, network.id)}_optimizer.pt'

    def _create_model_name(self, network: Network):
        return f'{os.path.join(self._folder_path, network.id)}_model.pt'

    def touch_folder(self, network_id: str):
        os.utime(self._folder_path)

    def optimize_for_loss(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

