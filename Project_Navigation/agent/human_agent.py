import time
from pynput import keyboard


class HumanAgent:
    def __init__(self):
        self.last_keyboard_action = None
        self.listener = keyboard.Listener(on_release=self.on_release)
        self.listener.start()

    def on_release(self, key):
        try:
            if key.char == 'w':
                print("Moving forwards.")
                self.last_keyboard_action = 0
            elif key.char == 's':
                print("Moving backwards.")
                self.last_keyboard_action = 1
            elif key.char == 'a':
                print("Moving left.")
                self.last_keyboard_action = 2
            elif key.char == 'd':
                print("Moving right.")
                self.last_keyboard_action = 3
        except AttributeError:
            pass

    def act(self, state=None):
        timeout_start = time.time()
        while self.last_keyboard_action is None and time.time() < timeout_start + 0.05:
            pass
        action = self.last_keyboard_action
        self.last_keyboard_action = None
        return action

    def step(self):
        raise RuntimeError("Human agent cannot take a training step.")
