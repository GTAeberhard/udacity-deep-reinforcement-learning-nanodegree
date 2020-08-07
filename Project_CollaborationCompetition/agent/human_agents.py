import time
from pynput import keyboard


class HumanAgents:
    def __init__(self):
        self.last_keyboard_action = [[0, 0], [0, 0]]
        self.listener = keyboard.Listener(on_release=self.on_release)
        self.listener.start()

    def on_release(self, key):
        try:
            if key.char == 'w':
                print("(Left Player) Moving up.")
                self.last_keyboard_action[1] = [0, 1]
            elif key.char == 's':
                print("(Left Player) Moving down.")
                self.last_keyboard_action[1] = [0, -1]
            elif key.char == 'a':
                print("(Left Player) Moving left.")
                self.last_keyboard_action[1] = [-0.25, 0]
            elif key.char == 'd':
                print("(Left Player) Moving right.")
                self.last_keyboard_action[1] = [0.25, 0]
        except AttributeError:
            if key == keyboard.Key.up:
                print("(Right Player) Moving up.")
                self.last_keyboard_action[0] = [0, 1]
            elif key == keyboard.Key.down:
                print("(Right Player) Moving down.")
                self.last_keyboard_action[0] = [0, -1]
            elif key == keyboard.Key.left:
                print("(Right Player) Moving left.")
                self.last_keyboard_action[0] = [0.25, 0]
            elif key == keyboard.Key.right:
                print("(Right Player) Moving right.")
                self.last_keyboard_action[0] = [-0.25, 0]

    def act(self, state=None):
        timeout_start = time.time()
        while self.last_keyboard_action is None and time.time() < timeout_start + 0.05:
            pass
        action = self.last_keyboard_action
        self.last_keyboard_action = [[0, 0], [0, 0]]
        return action

    def step(self, state, action, reward, next_state, done):
        raise RuntimeError("Human agents cannot take a training step.")