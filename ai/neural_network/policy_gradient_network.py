import os
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as Function
from torch.distributions import Multinomial

from ai.config.neural_net_config import Config, SGDConfig
from ai.lib.network_manager import Network, NetworkManager
from ai.lib.decider import Decider

dtype = torch.double
# device = torch.device("cpu")
device = torch.device("cuda:0")
verbose = False


@dataclass
class Round:
    value: torch.Tensor
    log_probability: torch.Tensor
    reward: float = 0


# TODO: Should obviously be split into two networks
class PolicyGradientNetwork(Decider):
    _base_network: Network
    _policy_gradient: Network
    _value_function: Network
    count = 0

    def __init__(self, config: Config):
        self._manager = NetworkManager(config)
        self.rounds: List[Round] = []
        self.discount_rate = config.options.get('discount_rate')

    def decide(self, choices: [[float]]) -> int:
        inputs = list(map(
            lambda choice: torch.FloatTensor(choice), choices))
        enhanced_features = list(map(
            lambda vec: self._base_network.model.forward(vec), inputs))
        action_features = list(map(
            lambda vec: self._policy_gradient.model.forward(vec.detach()),
            enhanced_features))

        # Get move
        probabilities = Function.softmax(torch.cat(list(action_features)))
        distribution = Multinomial(1, probabilities)
        move = distribution.sample()
        _, index_of_move = move.max(0)
        self.count += 1
        # if self.count % 1000 == 1:
        #     print(probabilities)

        # Expected reward
        expected_reward = self._value_function.model(
            enhanced_features[index_of_move])
        log_probability = distribution.log_prob(move)

        # Record estimate
        self.rounds.append(Round(
            value=expected_reward,
            log_probability=log_probability))

        # Return
        return index_of_move.item()

    def save(self):
        self._manager.save()

    # TODO: Change to reward vector to make generic
    def give_reward(self, reward: float):
        if len(self.rounds):
            self.rounds[-1].reward += reward
            for i, round_data in enumerate(reversed(self.rounds)):
                discounted_reward = reward * self.discount_rate ** i
                round_data.reward += discounted_reward

    def finish_episode(self):
        self._manager.zero_grad()
        self._value_function_loss.backward()
        self._pg_function_loss.backward()
        self._manager.step()
        self.rounds = []

    @property
    def _pg_function_loss(self):
        log_probabilities = torch.stack(list(
            round_data.log_probability
            for round_data in self.rounds))
        return torch.dot(-self._reward_tensor, log_probabilities)

    @property
    def _value_function_loss(self) -> torch.FloatTensor:
        value_estimate_tensor = torch.stack(list(
            round_data.value
            for round_data in self.rounds))
        return (self._reward_tensor - value_estimate_tensor).pow(2).sum()

    @property
    def _reward_tensor(self):
        return torch.FloatTensor(
            list(round_data.reward
            for round_data in self.rounds))

    @property
    def _value_function(self) -> Network:
        return self._manager.get(network_id='value_function')

    @property
    def _base_network(self) -> Network:
        return self._manager.get(network_id='base_network')

    @property
    def _policy_gradient(self) -> Network:
        return self._manager.get(network_id='policy_gradient')

