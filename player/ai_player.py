from itertools import chain
import random
from typing import Generic, TypeVar

from ai.features.feature_formatter import FeatureFormatter
from player.player import Player
from ai.lib.decider import Decider
from tetris.block import Block
from tetris.game_state import GameState, Cell
from tetris.move_list import Move

T = TypeVar('T')


class AIPlayer(Player, Generic[T]):

    game_count = 0
    mega_counter = 0
    move_count = 0

    @property
    def should_render(self) -> bool:
        return self.game_count % 50 == 1

    def __init__(
            self,
            neural_network: Decider,
            feature_formatter: FeatureFormatter[T]):
        self.neural_network = neural_network
        self.move_counter = 0
        self.formatter = feature_formatter

    def make_move(self, game_state: GameState, block: Block):
        choices = self.formatter.create_choices(game_state, block)
        self.move_count += 1
        if self.move_count % 5 == 0:
            return Move.down

        choice = self.neural_network.decide(choices=choices)
        if random.random() > 0.90:
            choice = int(random.random() * 5)
        self.every_now_and_then()
        return Move(choice)

    def every_now_and_then(self):
        self.move_counter = (self.move_counter + 1) % 2000
        if self.move_counter == 1:
            self.mega_counter += 1
        if self.mega_counter % 10 == 0:
            self.export()
            self.mega_counter += 1

    def export(self):
        self.neural_network.save()
        print('Exporting model')

    def give_reward(self, reward):
        self.neural_network.give_reward(reward)
        if reward == -100:
            # print('Learning...')
            self.neural_network.finish_episode()
