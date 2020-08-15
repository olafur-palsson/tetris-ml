from enum import Enum
from itertools import chain

from ai.agent.agent import Agent
from ai.neural_net_config import AgentConfig
from ai.neural_network.neural_network import NeuralNetwork
from ai.neural_network.policy_gradient_network import PolicyGradientNetwork
from game.block import Block
from game.game_state import GameState, Cell


class MoveList(Enum):
    down = 0
    right = 1
    left = 2
    rotate = 3


move_vectors = [
    [0, 0, 0],  # down
    [0, 0, 1],  # right
    [0, 1, 0],  # left
    [1, 0, 0]   # rotate
]


class NeuralNetworkAgent(Agent):

    def __init__(self, neural_network: NeuralNetwork):
        self.neural_network = neural_network

    def make_move(self, game_state: GameState, block: Block):
        features = self._convert_to_feature_vector(game_state, block)
        available_moves = self._create_move_vectors(features)
        choice = self.neural_network.choose(options=available_moves)
        return MoveList(choice)

    def give_reward(self, reward):
        self.neural_network.give_reward(reward)

    def _create_move_vectors(self, features: [float]) -> [float]:
        return [
            [*move, *features]
            for move in move_vectors]

    def _convert_to_feature_vector(
            self,
            game_state: GameState,
            current_block: Block):
        game_features = list(map(
            self._cell_to_float,
            self._flatten_vector(game_state.state)))
        current_block_features = list(map(
            self._bool_to_float,
            self._flatten_vector(current_block.state)))
        return [game_features, current_block_features]

    def _flatten_vector(self, vector):
        return list(chain(vector))

    def _cell_to_float(self, cell: Cell):
        return 0 if cell.empty else 1

    def _bool_to_float(self, boolean: bool):
        return 1 if boolean else 0

