from typing import List

from engine.engine import Engine
from player.player import Player
from tetris.tetris import Tetris
from tetris.tetris import GameState
from tetris.block import BlockFactory

ANALYTIC_BASE = 500


class Analytics:

    score = 0
    squares_filled = 0
    games = 0

    def print_analytics(self, score):
        metric = ANALYTIC_BASE if self.games > ANALYTIC_BASE else self.games
        print(
            f'{self.games} {metric}'
            
            f'\tScore: {round(self.score / metric, 3)} '
            f'\tFilled: {round(self.squares_filled / metric, 3)}'
            f'\tScore, current: {round(score, 3)}')

    def log_game(self, game_state: GameState, score: float):
        self.games += 1
        self.add_squares_filled(game_state)
        self.add_score(score)

    def add_squares_filled(self, game_state: GameState):
        squares_filled = 0
        for line in game_state.state:
            for cell in line:
                squares_filled += 1 if not cell.empty else 0
        if self.games > ANALYTIC_BASE:
            delta = squares_filled - self.squares_filled / ANALYTIC_BASE
            self.squares_filled += delta
        else:
            self.squares_filled += squares_filled

    def add_score(self, score: float):
        if self.games > ANALYTIC_BASE:
            delta = score - self.score / ANALYTIC_BASE
            self.score += delta
        else:
            self.score += score


class Game:

    def __init__(
            self,
            player: Player,
            block_factories: List[BlockFactory]):
        self.player = player
        self.block_factories = block_factories
        self.engine = Engine()
        self.analytics = Analytics()
        self.counter = 0

    def play(self):
        self.counter += 1
        initial_state = GameState()
        tetris = Tetris(
            block_factories=self.block_factories,
            initial_state=initial_state)
        self.player.game_count += 1
        total_score = 0
        while not tetris.is_lost():
            if self.player.should_render:
                self.render(tetris)
            move = self.player.make_move(
                tetris.game_state,
                tetris.current_block)
            score = tetris.handle_move(move)
            total_score += score
            if score:
                self.player.give_reward(score)
        # For losing. Bitch as hoe
        self.analytics.log_game(tetris.game_state, total_score)
        self.analytics.print_analytics(total_score)
        squares_filled = 0
        for line in tetris.game_state.state:
            for cell in line:
                squares_filled += 1 if not cell.empty else 0
        self.player.give_reward(-100)

    def render(self, tetris):
        self.engine.render_frame(tetris=tetris)

    @property
    def config(self):
        return {'HEIGHT': 20, 'WIDTH': 10}
