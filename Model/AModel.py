import torch
from os.path import join
from Config.AConfig import AConfig


class AModel(torch.nn.Module):
    def __init__(self, conf: AConfig):
        super().__init__()
        self.conf = conf
        self.device = None

    def save(self, save_dir):
        self.conf.save(join(save_dir, "model_conf.json"))
        torch.save(self.state_dict(), join(save_dir, "model_weight.bin"))

    def load(self, load_dir):
        self.load_state_dict(torch.load(join(load_dir, "model_weight.bin"), map_location="cpu"))

    def get_device(self):
        if self.device is None:
            for v in self.state_dict().values():
                self.device = v.device
                break
        return self.device
