from dataclasses import dataclass
from typing import List

from tetris.block import Block
from tetris.color import Color
from tetris.position import Position


@dataclass
class Cell:
    color: Color
    empty: bool


class GameState:
    width: int
    height: int
    state: List[List[Cell]]

    def __init__(self, height=20, width=10):
        self.height = height
        self.width = width
        self.state = [self.create_new_row() for i in range(height)]

    def clear_filled_rows(self):
        count = 0
        while (index := self.find_index_of_next_full_row()) is not None:
            popped = self.state.pop(index)
            row = self.create_new_row()
            self.state.append(row)
            count += 1
            if count > 3:
                raise Exception('This one is benign :D ')

    def print_state(self):
        for row in self.state:
            string = '\t'.join(map(lambda cell: cell.color.value, row))
            print(string)

    def create_new_row(self):
        return [Cell(color=Color.TRANSPARENT, empty=True)
                for _ in range(self.width)]

    def find_index_of_next_full_row(self) -> int:
        value = next(
            (
                i
                for i, row in enumerate(self.state)
                if self.row_is_full(row)
            ), None)
        return value

    @property
    def number_of_filled_rows(self):
        return sum(
            1
            for row in self.state
            if self.row_is_full(row))

    def row_is_empty(self, row):
        return not any(map(lambda cell: not cell.empty, row))

    def row_is_full(self, row):
        return all(map(lambda cell: not cell.empty, row))

    def lock_in_block(self, block: Block):
        for position in block.occupied_positions:
            self.state[position.y][position.x] = Cell(
                block.color, empty=False)

    def block_fits(self, block: Block):
        return all(
            self.position_fits(position)
            for position in block.occupied_positions)

    def position_fits(self, position: Position) -> bool:
        is_out_of_bounds = self.position_is_out_of_bounds(position)
        return (
            False
            if is_out_of_bounds
            else self.cell_is_empty(position))

    def cell_is_empty(self, position: Position) -> bool:
        cell = self.state[position.y][position.x]
        return cell.empty

    def position_is_out_of_bounds(self, position: Position) -> bool:
        has_negative = position.x < 0 or position.y < 0
        # too_high = position.y >= self.height
        too_sideways = position.x >= self.width
        return has_negative or too_sideways
