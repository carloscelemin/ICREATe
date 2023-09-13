from pyglet.window import key
"""
Class that obtains the human feedback from the computer's keyboard.
"""
class Keyboard:
    def __init__(self, env, config):
        env.unwrapped.viewer.window.on_key_press = self.key_press
        env.unwrapped.viewer.window.on_key_release = self.key_release
        self.FB_dict = [eval(f"key.{item.upper()}") for item in dict(config["FEEDBACK"]).keys()]
        self.FB_values = [int(item) for item in dict(config["FEEDBACK"]).values()]
        self.h = -1

    def key_press(self, k, mod):
        self.h = self.FB_values[ self.FB_dict.index(k) ] if k in self.FB_dict else -1

    def key_release(self, k, mod):
        self.h = -1

    def get_h(self):
        return self.h

    def ask_for_done(self):
        done = self.restart
        self.restart = False
        return done