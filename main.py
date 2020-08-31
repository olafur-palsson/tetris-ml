import argparse
import json

import yaml
from dacite import from_dict

from ai.config.neural_net_config import Config
from ai.neural_network.policy_gradient_network import PolicyGradientNetwork
from config import Level
from game import Game
from player.ai_player import AIPlayer
from player.human_player import HumanPlayer
from player.player import Player


def main():
    args = get_args()
    player = get_player(args)
    block_factories = get_level(args).value
    game = Game(
        block_factories=block_factories,
        player=player)
    while True:
        game.play()


def get_player(args) -> Player:
    if args.player == 'human':
        return HumanPlayer()
    network_path = args.network
    with open(network_path) as file:
        if network_path.endswith('json'):
            config = from_dict(Config, json.load(file))
        elif network_path.endswith('yaml'):
            config = from_dict(Config, yaml.load(file))
    network = PolicyGradientNetwork(config)
    return AIPlayer(network)


def get_level(args) -> Level:
    level = args.level.upper()
    return Level[level]


def get_args():
    parser = argparse.ArgumentParser(description='Tetris solution')
    parser.add_argument(
        '-p',
        '--player',
        help='one of { human, bot }',
        default='human'
    )
    parser.add_argument(
        '-l',
        '--level',
        help='one of { one, two, ..., eight, tetris }',
        default='tetris')
    parser.add_argument(
        '-n',
        '--network',
        help='path to a network'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
