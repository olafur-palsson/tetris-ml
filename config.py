from enum import Enum

from tetris.block import *

t = TBlockFactory()
i = IBlockFactory()
p = PBlockFactory()
l = LBlockFactory()
s = SBlockFactory()
z = ZBlockFactory()
square = SquareBlockFactory()
training = TrainingBlockFactory()


class Level(Enum):
    ONE = [training]
    TWO = [training, training, square]
    THREE = [square, training]
    FOUR = [l, p, square, training, training]
    FIVE = [l, p, i, square, training]
    SIX = [l, p, s, z, training, training, training]
    SEVEN = [l, p, s, z, square, training, training, training, training]
    EIGHT = [l, p, i, s, z, square, training, training]
    TETRIS = [t, i, p, l, s, z, square]
