from typing import List

from pynput.keyboard import Key

from engine.engine import Engine
from engine.input_handler import InputHandler
from game.tetris import Tetris
from game.tetris import GameState
from game.block import (
    Block,
    BlockFactory,
    TBlockFactory,
    IBlockFactory,
    PBlockFactory,
    LBlockFactory,
    SBlockFactory,
    ZBlockFactory
)


class Game:

    def __init__(self):
        initial_state = GameState()
        self.engine = Engine()
        self.input_handler = InputHandler()
        self.tetris = Tetris(
            block_factories=self.block_factories,
            initial_state=initial_state)
        self.start_game_loop()

    def start_game_loop(self):
        while not self.tetris.is_lost():
            self.engine.clear()
            self.engine.render_game_state(self.tetris.game_state)
            self.engine.render_current_block(self.tetris.current_block)
            key = self.input_handler.get_next_game_key()
            self.handle_key(key)

    def handle_key(self, key: Key):
        if key == Key.space:
            self.tetris.rotate_clockwise()
        elif key == Key.right:
            self.tetris.right()
        elif key == Key.left:
            self.tetris.left()
        elif key == Key.down:
            self.tetris.down()
        elif key == Key.enter:
            self.tetris.down()
            self.tetris.down()
            self.tetris.down()
            self.tetris.down()
            self.tetris.down()

    @property
    def config(self):
        return {'HEIGHT': 20, 'WIDTH': 10}

    @property
    def block_factories(self) -> List[BlockFactory]:
        return [
            TBlockFactory(),
            IBlockFactory(),
            PBlockFactory(),
            LBlockFactory(),
            SBlockFactory(),
            ZBlockFactory()
        ]


def main():
    Game()


if __name__ == '__main__':
    main()
