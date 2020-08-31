from typing import List

import torch


class Decider:

    def give_reward(self, reward):
        raise NotImplementedError()

    def decide(self, choices: List[List[torch.Tensor]]) -> int:
        raise NotImplementedError()

    def finish_episode(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()