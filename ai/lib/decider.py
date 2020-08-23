from typing import List

import torch

from ai.config.neural_net_config import NetworkConfig


class Decider:

    def give_reward(self, reward):
        raise NotImplementedError()

    def decide(self, options: List[List[torch.Tensor]]) -> int:
        raise NotImplementedError()

    def finish_episode(self):
        raise NotImplementedError()
