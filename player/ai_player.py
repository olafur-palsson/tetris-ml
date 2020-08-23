from itertools import chain

from player.player import Player
from ai.lib.decider import Decider
from game.block import Block
from game.game_state import GameState, Cell
from game.move_list import Move


class NeuralNetworkAgent(Player):

    def __init__(self, neural_network: Decider):
        self.neural_network = neural_network
        self.move_counter = 0

    def make_move(self, game_state: GameState, block: Block):
        features = self._convert_to_feature_vector(game_state, block)
        available_moves = self._create_move_vectors(features)
        choice = self.neural_network.decide(options=available_moves)
        self.export_intermittently()
        return Move(choice)

    def export_intermittently(self):
        self.move_counter = (self.move_counter + 1) % 50000
        if self.move_counter == 10000:
            self.export()

    def export(self):
        self.neural_network.export()
        print('Exporting model')

    def give_reward(self, reward):
        self.neural_network.give_reward(reward)
