from enum import Enum


class Move(Enum):
    down = 0
    right = 1
    left = 2
    rotate = 3

    @classmethod
    def move_vectors(cls):
        return [
            [0, 0, 0],  # down
            [0, 0, 1],  # right
            [0, 1, 0],  # left
            [1, 0, 0]   # rotate
        ]

