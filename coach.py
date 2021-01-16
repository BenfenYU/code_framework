from base_code import EasyDict, Visualizer


class Trainer:
    def __init__(self,wrapper):
        self.args = wrapper.args
        self.wrapper = wrapper
        self.visulizer = Visualizer(self.args)
        self.device = wrapper.device

    def train_config_a(self):
        print("training config_a ...")
        return

    def train_config_b(self):
        return

class Inferencer:
    def __init__(self,wrapper):
        self.args = wrapper.args
        self.wrapper = wrapper
        self.visualizer = Visualizer(self.args)
        self.device = wrapper.device

    def inference_config_a(self):
        pass
