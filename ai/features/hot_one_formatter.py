from typing import List

from ai.features.feature_formatter import FeatureFormatter, T
from tetris.block import Block
from tetris.game_state import GameState
from tetris.move_list import Move


class HotOneFormatter(FeatureFormatter[List[float]]):

    def create_choices(
            self,
            game_state: GameState,
            block: Block) -> List[List[float]]:
        features = self._convert_to_feature_vector(game_state, block)
        return self._create_move_vectors(features)

    def _create_move_vectors(self, features: [float]) -> [[float]]:
        return [
            [*move, *features]
            for move in Move.move_vectors()]

    def _convert_to_feature_vector(
            self,
            game_state: GameState,
            current_block: Block):
        game_features = list(map(
            self._cell_to_float,
            self._flatten_vector(game_state.state)))
        current_block_features = list(map(
            self._bool_to_float,
            self._flatten_vector(current_block.state)))
        x_position_features = [0 for i in range(12)]
        x_position_features[current_block.x] = 1
        y_position_features = [0 for i in range(20)]
        y_position_features[current_block.y] = 1

        return [
            *game_features,
            *current_block_features,
            *x_position_features,
            *y_position_features
        ]
