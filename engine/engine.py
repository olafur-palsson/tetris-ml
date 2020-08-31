from engine.graphics import Polygon, Point, GraphWin, GraphicsObject
from tetris.block import Block
from tetris.color import Color
from tetris.game_state import GameState
from tetris.position import Position


class Engine:

    def __init__(self, height=600, width=300):
        self.window = GraphWin('Tetris', width, height)
        self.height = height
        self.width = width
        self.square_size = width / 10
        self.drawing_cache: [GraphicsObject] = []

    def render_frame(self, tetris):
        self.clear()
        self.render_game_state(tetris.game_state)
        self.render_current_block(tetris.current_block)

    def clear(self):
        for graphics_object in self.drawing_cache:
            graphics_object.undraw()
        self.drawing_cache = []

    def render_current_block(self, block: Block):
        for position in block.occupied_positions:
            self.draw_square(position, block.color)

    def render_game_state(self, game_state: GameState):
        for y, row in enumerate(game_state.state):
            for x, cell in enumerate(row):
                if cell.color != Color.TRANSPARENT:
                    self.draw_square(
                        position=Position(x, y),
                        color=cell.color)

    def draw_square(self, position: Position, color: Color):
        vertices = self.create_vertices_for_square(position)
        square = Polygon(vertices)
        square.setFill(color.value)
        square.setWidth(0)
        square.draw(self.window)
        self.drawing_cache.append(square)

    def create_vertices_for_square(self, position: Position):
        size = self.width / 10
        adjusted = self.game_position_to_pixels(position)
        return [
            Point(adjusted.x,        self.height - adjusted.y),
            Point(adjusted.x + size, self.height - adjusted.y),
            Point(adjusted.x + size, self.height - (adjusted.y + size)),
            Point(adjusted.x,        self.height - (adjusted.y + size)),
        ]

    def game_position_to_pixels(self, position: Position) -> Position:
        return Position(
            x=round(position.x * self.square_size),
            y=round(position.y * self.square_size))
