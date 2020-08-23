import random
from player.player import Player
from game.move_list import Move


class RandomAgent(Player):

    def make_move(self, state, current_block):
        return random.choice(list(Move))

    def give_reward(self, reward):
        pass
