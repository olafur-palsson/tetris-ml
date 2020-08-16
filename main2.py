import json
from typing import List

from dacite import from_dict
import time

from ai.agent.neural_network_agent import NeuralNetworkAgent
from ai.config.neural_net_config import AgentConfig
from ai.neural_network.policy_gradient_network import PolicyGradientNetwork
from engine.engine import Engine
from game.move_list import Move
from game.tetris import Tetris
from game.tetris import GameState
from game.block import (
    BlockFactory,
    TBlockFactory,
    IBlockFactory,
    PBlockFactory,
    LBlockFactory,
    SBlockFactory,
    ZBlockFactory, SquareBlockFactory, TrainingBlockFactory
)


class Game:

    def __init__(self, agent_config: AgentConfig, should_display: bool = False):
        self.should_display = should_display
        self.engine = Engine()
        neural_network = PolicyGradientNetwork(agent_config)
        self.agent = NeuralNetworkAgent(neural_network)
        self.play_forever()

    def play_one_game(self):
        initial_state = GameState()
        tetris = Tetris(
            block_factories=self.block_factories,
            initial_state=initial_state)

        current_height = 0

        while not tetris.is_lost():
            if self.should_display:
                time.sleep(0.05)
                self.engine.render_frame(tetris)
            move = self.agent.make_move(
                tetris.game_state,
                tetris.current_block)
            reward = self.execute_move(tetris, move)
            height = tetris.block_height
            punishment = current_height - height
            current_height = height

            if reward:
                print(f'Reward! {reward}')
                self.agent.give_reward(reward)
            elif punishment:
                print(f'Punish! :) {punishment}')
                self.agent.give_reward(-punishment)
        return tetris.score

    def play_forever(self):
        while True:
            self.play_one_game()

    def execute_move(self, tetris: Tetris, move: Move) -> int:
        if move == Move.rotate:
            tetris.rotate_clockwise()
        elif move == Move.right:
            tetris.right()
        elif move == Move.left:
            tetris.left()
        elif move == Move.down:
            return tetris.down()
        return 0

    @property
    def config(self):
        return {'HEIGHT': 20, 'WIDTH': 10}

    @property
    def block_factories(self) -> List[BlockFactory]:
        return [
            TrainingBlockFactory()
            # SquareBlockFactory(),
            # TBlockFactory(),
            # IBlockFactory(),
            # PBlockFactory(),
            # LBlockFactory(),
            # SBlockFactory(),
            # ZBlockFactory()
        ]


def parse_config(path_to_config):
    with open(path_to_config) as file:
        config = json.load(file)
        print(json.dumps(config, indent=4))
        return from_dict(AgentConfig, config)


def main():
    agent_config = parse_config('./ai/config/agent_config.json')
    Game(agent_config=agent_config)


if __name__ == '__main__':
    main()
