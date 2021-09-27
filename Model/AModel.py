import torch
from os.path import join
from Config.AConfig import AConfig


class AModel(torch.nn.Module):
    def __init__(self, conf: AConfig):
        super().__init__()
        self.conf = conf

    def save(self, save_dir):
        self.conf.save(join(save_dir, "model_conf.json"))
        torch.save(self.state_dict(), join(save_dir, "model_weight.bin"))

    @classmethod
    def load(cls, load_dir):
        conf = AConfig.load(join(load_dir, "model_conf.json"))
        model = cls(conf)
        model.load_state_dict(torch.load(join(load_dir, "model_weight.bin"), map_location="cpu"))
        return model

    def get_device(self):
        for v in self.state_dict().values():
            return v.device
