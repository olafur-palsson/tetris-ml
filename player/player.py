#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from game.block import Block
from game.game_state import GameState
from game.move_list import Move


class Player:

    def make_move(self, state, current_block) -> Move:
        raise NotImplementedError()

    def give_reward(self, reward):
        raise NotImplementedError()

    def _create_move_vectors(self, features: [float]) -> [float]:
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

    def _flatten_vector(self, vector):
        return list(chain(*vector))

    def _cell_to_float(self, cell: Cell):
        return 0 if cell.empty else 1

    def _bool_to_float(self, boolean: bool):
        return 1 if boolean else 0

