from typing import Iterable, List
from dataclasses import dataclass

from tetris.color import Color
from tetris.position import Position


@dataclass
class Block:
    state: Iterable[Iterable[bool]]
    x: int
    y: int
    color: Color

    def rotate_clockwise(self):
        rows_reversed = self.state[::-1]
        self.state = list(zip(*rows_reversed))

    def rotate_counter_clockwise(self):
        self.rotate_clockwise()
        self.rotate_clockwise()
        self.rotate_clockwise()

    @property
    def occupied_positions(self) -> Iterable[Position]:
        return [
            Position(self.x - offset_x, self.y - offset_y)
            for offset_y, row in enumerate(self.state)
            for offset_x, value in enumerate(row)
            if value
        ]

    def move_left(self):
        self.x -= 1

    def move_right(self):
        self.x += 1

    def move_down(self):
        self.y -= 1

    def move_up(self):
        self.y += 1


class BlockFactory:

    def create_block(self) -> Block:
        positions = list(
            map(lambda row: self.symbol_list_to_bool_list(row),
                self.template))
        return Block(
            state=positions,
            x=5,
            y=19,
            color=self.color)

    def symbol_list_to_bool_list(self, symbol_list):
        return list(
            map(lambda symbol: symbol == 'x', symbol_list))

    @property
    def template(self) -> List[List[str]]:
        raise NotImplementedError()

    @property
    def color(self) -> Color:
        raise NotImplementedError()


class PBlockFactory(BlockFactory):

    @property
    def color(self) -> Color:
        return Color.ORANGE

    @property
    def template(self):
        return [['.', 'x', 'x', '.'],
                ['.', 'x', '.', '.'],
                ['.', 'x', '.', '.'],
                ['.', '.', '.', '.']]


class SquareBlockFactory(BlockFactory):

    @property
    def color(self) -> Color:
        return Color.YELLOW

    @property
    def template(self) -> List[List[str]]:
        return [['.', '.', '.', '.'],
                ['.', 'x', 'x', '.'],
                ['.', 'x', 'x', '.'],
                ['.', '.', '.', '.']]


class TrainingBlockFactory(BlockFactory):

    @property
    def color(self) -> Color:
        return Color.YELLOW

    @property
    def template(self) -> List[List[str]]:
        return [['.', '.', '.', '.'],
                ['.', '.', '.', '.'],
                ['.', 'x', 'x', '.'],
                ['.', '.', '.', '.']]


class LBlockFactory(BlockFactory):

    @property
    def color(self) -> Color:
        return Color.BLUE

    @property
    def template(self):
        return [['.', 'x', '.', '.'],
                ['.', 'x', '.', '.'],
                ['.', 'x', 'x', '.'],
                ['.', '.', '.', '.']]


class SBlockFactory(BlockFactory):

    @property
    def color(self) -> Color:
        return Color.GREEN

    @property
    def template(self):
        return [['.', 'x', '.', '.'],
                ['.', 'x', 'x', '.'],
                ['.', '.', 'x', '.'],
                ['.', '.', '.', '.']]


class TBlockFactory(BlockFactory):

    @property
    def color(self) -> Color:
        return Color.PURPLE

    @property
    def template(self):
        return [['.', 'x', '.', '.'],
                ['.', 'x', 'x', '.'],
                ['.', 'x', '.', '.'],
                ['.', '.', '.', '.']]


class ZBlockFactory(BlockFactory):

    @property
    def color(self) -> Color:
        return Color.RED

    @property
    def template(self):
        return [['.', '.', 'x', '.'],
                ['.', 'x', 'x', '.'],
                ['.', 'x', '.', '.'],
                ['.', '.', '.', '.']]


class IBlockFactory(BlockFactory):

    @property
    def color(self) -> Color:
        return Color.TEAL

    @property
    def template(self):
        return [['.', 'x', '.', '.'],
                ['.', 'x', '.', '.'],
                ['.', 'x', '.', '.'],
                ['.', 'x', '.', '.']]
