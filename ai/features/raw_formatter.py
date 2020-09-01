from dataclasses import dataclass
from typing import List

from ai.features.feature_formatter import FeatureFormatter, T
from tetris.block import Block
from tetris.game_state import GameState
from tetris.move_list import Move


@dataclass
class RawFeatures:
    block: List[List[float]]
    board: List[List[float]]
    position: List[float]
    move: List[float]


class RawFormatter(FeatureFormatter[RawFeatures]):

    def create_choices(
            self,
            game_state: GameState,
            block: Block) -> List[RawFeatures]:
        board_features = self._2d_cell_to_2d_float(game_state.state)
        block_features = self._2d_bool_to_2d_float(block.state)
        return [
            RawFeatures(
                board=board_features,
                block=block_features,
                position=self._position_vector(block),
                move=move_vector)
            for move_vector in Move.move_vectors()
        ]
