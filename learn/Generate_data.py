import numpy as np
import torch

from utils.Config import Config
from benchmarks.Examplers import Zone, Example


class Data:
    def __init__(self, config: Config):
        self.config = config
        self.ex = config.EXAMPLE
        self.n = self.ex.n
        self.batch_size = self.config.BATCH_SIZE

    def get_data(self, zone: Zone, batch_size):
        global s
        if zone.shape == 'box':
            s = np.random.rand(batch_size, self.n)
            center = (zone.low + zone.up) / 2
            s = s * (zone.up - zone.low) + center

        elif zone.shape == 'ball':
            s = np.random.randn(batch_size, self.n)
            s = np.array([e / np.sqrt(sum(e ** 2)) * np.sqrt(zone.r) * np.random.random() ** (1 / self.n) for e in s])
            s = s + zone.center

        # from matplotlib import pyplot as plt
        # plt.plot(s[:, :1], s[:, -1], '.')
        # plt.gca().set_aspect(1)
        # plt.show()
        return torch.Tensor(s)

    def complement_of_target(self):
        domain = self.get_data(self.ex.l, self.batch_size)
        if self.ex.target.shape == 'box':
            pass
        else:
            pass

    def x2dotx(self, X, f):
        f_x = []
        for x in X:
            f_x.append([f[i](x) for i in range(self.n)])
        return torch.Tensor(f_x)

    def generate_data(self):
        batch_size = self.config.BATCH_SIZE
        target = self.get_data(self.ex.target, batch_size)
        l = self.get_data(self.ex.l1, batch_size)

        l1_dot = self.x2dotx(l, self.ex.f)

        center = None
        return target, l, l1_dot, center
