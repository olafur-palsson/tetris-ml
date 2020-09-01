from itertools import chain
from typing import List, TypeVar, Generic, Iterable

from tetris.block import Block
from tetris.game_state import GameState, Cell

T = TypeVar('T')


class FeatureFormatter(Generic[T]):

    def create_choices(self, game_state: GameState, block: Block) -> List[T]:
        raise NotImplementedError()

    def _flatten_vector(self, vector):
        return list(chain(*vector))

    def _cell_to_float(self, cell: Cell):
        return 0 if cell.empty else 1

    def _bool_to_float(self, boolean: bool):
        return 1 if boolean else 0

    def _2d_bool_to_2d_float(
            self,
            input_mat: Iterable[Iterable[bool]]) -> List[List[float]]:
        return [
            [1 if el else 0 for el in line]
            for line in input_mat
        ]

    def _position_vector(self, block: Block) -> List[float]:
        x_position_features = [0 for i in range(12)]
        x_position_features[block.x] = 1
        y_position_features = [0 for i in range(20)]
        y_position_features[block.y] = 1
        return [*x_position_features, *y_position_features]

    def _2d_cell_to_2d_float(
            self,
            input_mat: List[List[Cell]]) -> List[List[float]]:
        return [
            [0 if el.empty else 1 for el in line]
            for line in input_mat
        ]
