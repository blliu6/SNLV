import numpy as np
import sympy as sp
import torch

from utils.Config import Config
from benchmarks.Examplers import Zone, get_example_by_name


def x2dotx(n, X, f, u):
    f_x = []
    x_ = sp.symbols([f'x{i + 1}' for i in range(n)])
    f_u = [sp.lambdify(x_, u[i]) for i in range(len(u))]
    res_u = []
    for x in X:
        ans = []
        for fun in f_u:
            ans.append(fun(*x))
        res_u.append(np.array(ans))
    res_u = np.array(res_u)

    for a, b in zip(X, res_u):
        f_x.append([f[i](a, b) for i in range(n)])
    return np.array(f_x)


class Data:
    def __init__(self, config: Config):
        self.config = config
        self.ex = config.EXAMPLE
        self.n = config.EXAMPLE.n
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
        self.draw(data)
        return data

    def generate_data(self):
        batch_size = self.batch_size
        target = self.ex.target
        target_data = self.complement_of_target()
        l = self.get_data(self.ex.local, batch_size)
        l1_dot = x2dotx(self.n, l, self.ex.f, self.config.controller)
        center = np.array([target.center] * batch_size)
        return torch.Tensor(target_data), torch.Tensor(l), torch.Tensor(l1_dot), torch.Tensor(center)


if __name__ == '__main__':
    ex = get_example_by_name('test')
    config = Config()
    config.EXAMPLE = ex
    da = Data(config)
    # da.complement_of_target()
    da.generate_data()
