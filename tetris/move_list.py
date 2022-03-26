from enum import Enum


class Move(Enum):
    noop = 0
    down = 1
    right = 2
    left = 3
    rotate = 4

    @classmethod
    def move_vectors(cls):
        return [
            [0, 0, 0, 0],  # no-op
            [0, 0, 0, 1],  # down
            [0, 0, 1, 0],  # right
            [0, 1, 0, 0],  # left
            [1, 0, 0, 0]   # rotate
        ]

