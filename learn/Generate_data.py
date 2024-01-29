import numpy as np
import torch

from utils.Config import Config
from benchmarks.Examplers import Zone, get_example_by_name


class Data:
    def __init__(self, config: Config):
        self.config = config
        self.ex = config.EXAMPLE
        self.n = self.ex.n
        self.batch_size = self.config.BATCH_SIZE

    def get_data(self, zone: Zone, batch_size):
        global s
        if zone.shape == 'box':
            s = np.random.rand(batch_size, self.n) - 0.5
            s = s * (zone.up - zone.low) + zone.center

        elif zone.shape == 'ball':
            s = np.random.randn(batch_size, self.n)
            s = np.array([e / np.sqrt(sum(e ** 2)) * np.sqrt(zone.r) * np.random.random() ** (1 / self.n) for e in s])
            s = s + zone.center

        # from matplotlib import pyplot as plt
        # plt.plot(s[:, :1], s[:, -1], '.')
        # plt.gca().set_aspect(1)
        # plt.show()
        return s

    def check_box(self, low, up, data):
        vis = True
        for i in range(len(low)):
            vis = vis and low[i] <= data[i] <= up[i]
            if not vis:
                return True
        return False

    def check_ball(self, center, r, data):
        if sum((data - center) ** 2) > r:
            return True
        return False

    def draw(self, s):
        from matplotlib import pyplot as plt
        plt.plot(s[:, :1], s[:, -1], '.')
        plt.gca().set_aspect(1)
        plt.show()

    def complement_of_target(self):
        target = self.ex.target
        domain = self.get_data(self.ex.local, 2 * self.batch_size)
        if target.shape == 'box':
            domain = [e for e in domain if self.check_box(target.low, target.up, e)]
            times = 2
            enhance_data = (np.random.rand(self.batch_size, self.n) - 0.5) * times
            res = []
            low, up = np.array([-0.5] * self.n), np.array([0.5] * self.n)
            for e in enhance_data:
                if self.check_box(low, up, e):
                    res.append(np.clip(e, -0.5, 0.5))
            enhance_data = np.array(res)
            enhance_data = enhance_data * (target.up - target.low) + target.center
        else:
            domain = [e for e in domain if self.check_ball(target.center, target.r, e)]

            enhance_data = np.random.randn(self.batch_size, self.n)
            enhance_data = np.array([e / np.sqrt(sum(e ** 2)) * np.sqrt(target.r) for e in enhance_data])
            enhance_data = enhance_data + target.center
        domain = np.array(domain)
        data = np.concatenate((domain, enhance_data), axis=0)
        # self.draw(data)
        return data

    def x2dotx(self, X, f):
        f_x = []
        for x in X:
            f_x.append([f[i](x) for i in range(self.n)])
        return np.array(f_x)

    def generate_data(self):
        batch_size = self.config.BATCH_SIZE
        target = self.ex.target
        target_data = self.complement_of_target()
        l = self.get_data(self.ex.local, batch_size)
        l1_dot = self.x2dotx(l, self.ex.f)
        center = np.array([target.center] * batch_size)
        return target_data, l, l1_dot, center


if __name__ == '__main__':
    ex = get_example_by_name('test')
    config = Config()
    config.EXAMPLE = ex
    da = Data(config)
    # da.complement_of_target()
    da.generate_data()
