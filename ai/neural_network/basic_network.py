#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This *IS* the neural network under consideration.
"""
import torch
import torch.nn as nn
from ai.neural_net_config import AgentConfig

dtype = torch.double
device = torch.device("cpu")
device = torch.device("cuda:0")  # Uncomment this to run on GPU


class BasicNeuralNetwork:

    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config
        self.predicted_rewards = []
        self.model = self._create_network_layers()
        self.optimizer = self._create_optimizer()

    def _create_network_layers(self):
        return nn.Sequential(*self._make_layers())

    def _create_optimizer(self):
        return torch.optim.SGD(
            self.model.parameters(),
            momentum=self.agent_config.sgd.momentum,
            lr=self.agent_config.sgd.momentum)

    def _make_layers(self, config):
        layer_widths = self._determine_layer_widths(config)
        # Total number of layers
        n = len(layer_widths)
        layers = []
        for i in range(n - 1):
            input_width = layer_widths[i]
            output_width = layer_widths[i + 1]
            layers = [*layers, nn.Linear(input_width, output_width)]
        return layers

    def _determine_layer_widths(self):
        return [
            self.agent_config.nn.input_layer,
            *self.agent_config.nn.hidden_layers,
            self.agent_config.nn.output_layer
        ]

    def run_decision(self, board_features, save_predictions=True):
        prediction = self.model(board_features)
        if save_predictions:
            self.predicted_rewards = torch.cat(
                (self.predicted_rewards, prediction.double()))
        return prediction

    def predict(self, input_features):
        with torch.no_grad():
            return self.model(input_features)

    def manually_reset_grad(self):
        self.optimizer.zero_grad()

    # for use in pub_stomper_policy gradient
    def manually_update_weights_of_network(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    # TODO: Refactor this shit, needs
    def calculate_loss_vector(self, reward_vector):
        temporal_delay = self.agent_config.nn.temporal_delay
        with torch.no_grad():
            for i in range(len(self.predicted_rewards)):
                index_of_last_prediction_rewarded = (
                        len(self.predicted_rewards) - temporal_delay)
                if i == index_of_last_prediction_rewarded:
                    break
                reward_vector[i] = self.predicted_rewards[i + temporal_delay]

    def evaluate(self, possible_boards):
        # possible_boards -> neural network -> sigmoid -> last_layer_sigmoid
        last_layer_outputs = self.run_through_neural_network(possible_boards)
        # last_layer_sigmoid = list(map(lambda x: x.sigmoid(), last_layer_outputs))

        # Decide move and save log_prob for backward
        # We make sure not to affect the value fn with .detach()

        probs = self.pg_plugin._softmax(last_layer_outputs)
        distribution = Multinomial(1, probs)
        move = distribution.sample()
        self.saved_log_probabilities.append(distribution.log_prob(move))

        _, move = move.max(0)
        # calculate the value estimation and save for backward
        value_estimate = self.pg_plugin.value_model(last_layer_outputs[move])
        self.saved_value_estimations.append(value_estimate)
        return move

    def give_reward_to_nn(self, reward):
        actual_rewards = self.create_reward_vector()
        loss = (self.predicted_rewards - actual_rewards).pow(2).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.reset_predicted_rewards()

    def reset_predicted_rewards(self):
        self.predicted_rewards = torch.empty(0, dtype=dtype, requires_grad=True)

    def create_reward_vector(self, reward):
        episode_length = len(self.predicted_rewards)
        return reward * torch.ones(
            episode_length,
            dtype=dtype,
            requires_grad=False)

