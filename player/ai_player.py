from itertools import chain
import random

from player.player import Player
from ai.lib.decider import Decider
from tetris.block import Block
from tetris.game_state import GameState, Cell
from tetris.move_list import Move


class AIPlayer(Player):

    game_count = 0
    mega_counter = 0

    @property
    def should_render(self) -> bool:
        return not (self.game_count % 50)

    def __init__(self, neural_network: Decider):
        self.neural_network = neural_network
        self.move_counter = 0

    def make_move(self, game_state: GameState, block: Block):
        features = self._convert_to_feature_vector(game_state, block)
        available_moves = self._create_move_vectors(features)
        if random.random() > 0.95:
            return Move.down
        choice = self.neural_network.decide(choices=available_moves)
        self.every_now_and_then()
        return Move(choice)

    def every_now_and_then(self):
        self.move_counter = (self.move_counter + 1) % 2000
        if self.move_counter == 250:
            self.mega_counter += 1
            print('Learning and exporting')
            self.neural_network.finish_episode()
        if self.mega_counter % 10 == 0:
            self.export()
            self.mega_counter += 1

    def export(self):
        self.neural_network.save()
        print('Exporting model')

    def give_reward(self, reward):
        self.neural_network.give_reward(reward)
