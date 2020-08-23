from pynput import keyboard
from pynput.keyboard import Key


class InputHandler:

    def __init__(self):
        self.key_pressed = None

    def get_next_game_key(self):
        while True:
            key = self._get_input_sync()
            if self._is_game_key(key):
                break
        return key

    def _is_game_key(self, key: Key):
        return key in [
            Key.esc,
            Key.down,
            Key.left,
            Key.right,
            Key.space,
            Key.enter
        ]

    def _get_input_sync(self):
        with keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        ) as listener:
            listener.join()
            return self.key_pressed

    def _on_press(self, key):
        self.key_pressed = key

    def _on_release(self, _):
        return False
