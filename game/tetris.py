from typing import List
import random

from game.block import BlockFactory, Block
from game.game_state import GameState


class Tetris:
    current_block: Block
    block_factories: List[BlockFactory]

    def __init__(
            self,
            block_factories: List[BlockFactory],
            initial_state: GameState):
        self.block_factories = block_factories
        self.game_state = initial_state
        self.current_block = self.generate_block()
        self.upcoming_block = self.generate_block()

    def generate_block(self) -> Block:
        factory = random.choice(self.block_factories)
        return factory.create_block()

    def is_lost(self) -> bool:
        return not self.game_state.block_fits(self.current_block)

    def down(self):
        self.current_block.move_down()
        if not self.game_state.block_fits(self.current_block):
            self.current_block.move_up()
            score = self.place_block()
            self.finish_round()
            return score
        else:
            return 0

    def place_block(self):
        self.game_state.lock_in_block(self.current_block)
        number_of_filled_rows = self.game_state.number_of_filled_rows
        self.game_state.clear_filled_rows()
        print(number_of_filled_rows)
        return number_of_filled_rows ** 2

    def finish_round(self):
        self.current_block = self.upcoming_block
        self.upcoming_block = self.generate_block()

    def left(self):
        self.current_block.move_left()
        if not self.game_state.block_fits(self.current_block):
            self.current_block.move_right()

    def right(self):
        self.current_block.move_right()
        if not self.game_state.block_fits(self.current_block):
            self.current_block.move_left()

    def rotate_clockwise(self):
        self.current_block.rotate_clockwise()
        if not self.game_state.block_fits(self.current_block):
            self.current_block.rotate_counter_clockwise()

    def rotate_counter_clockwise(self):
        self.current_block.rotate_counter_clockwise()
        if not self.game_state.block_fits(self.current_block):
            self.current_block.rotate_clockwise()
