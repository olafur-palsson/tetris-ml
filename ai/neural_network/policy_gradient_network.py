from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as Function
from torch.distributions import Multinomial

from ai.neural_net_config import AgentConfig, SGDConfig
from ai.neural_network.neural_network import NeuralNetwork

dtype = torch.double
# device = torch.device("cpu")
device = torch.device("cuda:0")


@dataclass
class Estimate:
    value: float
    log_probability: float


# TODO: Should obviously be split into two networks
class PolicyGradientNetwork(NeuralNetwork):
    estimates: [Estimate]

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config
        self.softmax_function = nn.Softmax()
        self.value_model = self._create_value_function_model(
            agent_config.nn.output_layer)
        self.value_optimizer = self._create_value_function_optimizer(
            agent_config.sgd)
        self.pg_optimizer = self._create_policy_gradient_optimizer(
            agent_config.sgd)
        self.pg_model = self._create_policy_gradient_model(
            agent_config.nn.output_layer)

        self.estimates = []

    def choose(self, options) -> int:
        probabilities = self._softmax(options)
        distribution = Multinomial(1, probabilities)
        move = distribution.sample()
        _, index_of_move = move.max(0)
        value_estimate = self.pg_plugin.value_function(options[index_of_move])
        log_probability = distribution.log_prob(move)
        self.estimates.append(Estimate(
            value=value_estimate,
            log_probability=log_probability))
        return index_of_move

    def _softmax(self, input_vectors):
        scores = list(map(
            lambda vector: self.pg_model(vector.detach()), input_vectors))
        return Function.softmax(torch.cat(scores))

    def value_function(self, input_vector):
        detached_input = input_vector.detach()
        return self.value_model.apply(detached_input)

    @property
    def _value_estimates(self):
        return list(
            map(lambda estimate: estimate.value, self.estimates))

    @property
    def _log_probability_estimates(self):
        return list(
            map(lambda estimate: estimate.log_probability, self.estimates))

    # TODO: Change to reward vector to make generic
    def give_reward(self, reward: float):
        loss = self._calculate_value_function_loss(reward)
        self._adjust_value_function_wrt_loss(loss)
        loss2 = self._calculate_pg_function_loss(reward)
        self._adjust_pg_function_wrt_loss(loss2)
        self.estimates = []

    def _calculate_pg_function_loss(self, reward):
        rewards = torch.ones(len(self.estimates)) * reward
        log_probs = torch.stack(self._log_probability_estimates)
        return torch.dot(-rewards, log_probs)

    def _calculate_value_function_loss(self, reward):
        values = torch.stack(self._value_estimates).squeeze()
        targets = self._create_targets(values, reward)
        return (targets - values).pow(2).sum()

    def _create_targets(self, value_vector, reward):
        temporal_delay = self.agent_config.nn.temporal_delay
        values_without_gradients = value_vector.detach()
        return torch.cat(
            values_without_gradients[temporal_delay:],
            torch.ones(temporal_delay) * reward)

    def _adjust_value_function_wrt_loss(self, loss):
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def _adjust_pg_function_wrt_loss(self, loss):
        self.pg_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def _reset_gradients(self):
        self.pg_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

    def _create_policy_gradient_model(self, output_layer_width: int):
        return nn.Sequential(
            nn.Linear(output_layer_width, 100),
            nn.Linear(100, 1))

    def _create_policy_gradient_optimizer(self, sgd_config: SGDConfig):
        return torch.optim.SGD(
            self.pg_model.parameters(),
            momentum=sgd_config.momentum,
            lr=sgd_config.learning_rate)

    def _create_value_function_optimizer(self, sgd_config: SGDConfig) -> torch.optim.SGD:
        return torch.optim.SGD(
            self.value_model.parameters(),
            momentum=sgd_config.momentum,
            lr=sgd_config.learning_rate)

    def _create_value_function_model(self, output_layer_width: int):
        return nn.Sequential(nn.Linear(output_layer_width, 1))
