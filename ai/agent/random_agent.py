import random
from ai.agent.agent import Agent
from game.move_list import Move


class RandomAgent(Agent):

    def make_move(self, state, current_block):
        return random.choice(list(Move))

    def give_reward(self, reward):
        pass
