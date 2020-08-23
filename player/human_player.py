from engine.input_handler import InputHandler
from game.move_list import Move
from player.player import Player
from pynput.keyboard import Key


class HumanPlayer(Player):
    __key_move_map = {
        Key.esc: Exception('Force exit'),
        Key.down: Move.down,
        Key.left: Move.left,
        Key.right: Move.right,
        Key.space: Move.rotate
    }

    def __init__(self):
        self.input_handler = InputHandler()

    def make_move(self, state, current_block) -> Move:
        move_key = self.input_handler.get_next_game_key()
        move = self.__key_move_map[move_key]
        if isinstance(move, Exception):
            raise move
        else:
            return move

    def give_reward(self, reward):
        pass
