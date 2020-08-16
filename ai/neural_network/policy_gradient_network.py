from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as Function
from torch.distributions import Multinomial

from ai.config.neural_net_config import AgentConfig, SGDConfig
from ai.neural_network.neural_network import NeuralNetwork

dtype = torch.double
# device = torch.device("cpu")
device = torch.device("cuda:0")
verbose = False


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
        self.pg_model = self._create_policy_gradient_model(
            agent_config.nn.output_layer)
        self.pg_optimizer = self._create_policy_gradient_optimizer(
            agent_config.sgd)

        self.estimates = []

    def print_average_abs_layer_weight(self, layer):
        layer_weight = abs(layer.weight.data).tolist()[0]
        print(f'Average abs layer weight', sum(layer_weight) / len(layer_weight))

    def choose(self, options) -> int:
        if verbose:
            self.print_average_abs_layer_weight(self.value_model[0])

        input_vectors = list(
            map(lambda option: torch.FloatTensor(option), options))
        probabilities = self._softmax(input_vectors)
        distribution = Multinomial(1, probabilities)
        move = distribution.sample()
        _, index_of_move = move.max(0)
        value_estimate = self.value_function(input_vectors[index_of_move])
        log_probability = distribution.log_prob(move)
        self.estimates.append(Estimate(
            value=value_estimate,
            log_probability=log_probability))
        return index_of_move.item()

    def _softmax(self, input_vectors):
        scores = list(map(
            lambda vector: self.pg_model(vector.detach()), input_vectors))
        return Function.softmax(torch.cat(scores))

    def value_function(self, input_vector):
        detached_input = input_vector.detach()
        return self.value_model(detached_input)

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
        log_probabilities = torch.stack(self._log_probability_estimates)
        return torch.dot(-rewards, log_probabilities)

    def _calculate_value_function_loss(self, reward):
        values = torch.stack(self._value_estimates).squeeze()
        episode_length = len(self._value_estimates)
        targets = self._create_targets(reward, episode_length)
        return (targets - values).pow(2).sum()

    def _create_targets(self, reward, episode_length):
        discount_constant = 0.9
        discount_vector = [
            reward * discount_constant ** i
            for i in range(episode_length)
        ]
        return torch.FloatTensor(discount_vector)

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
            nn.Linear(output_layer_width, 100, bias=True),
            nn.Linear(100, 1, bias=True))

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
