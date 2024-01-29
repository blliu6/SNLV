import numpy as np
from constants import *

class Zone:
    def __init__(self, shape: str, low=None, up=None, center=None, r=None):
        self.shape = shape
        if shape == 'ball':
            self.center = np.array(center, dtype=np.float32)
            self.r = r  # radius squared
        elif shape == 'box':
            self.low = np.array(low, dtype=np.float32)
            self.up = np.array(up, dtype=np.float32)
            self.center = (self.low + self.up) / 2
        else:
            raise ValueError(f'There is no area of such shape!')


class Example:
    def __init__(self, n, local, target, f, name):
        self.n = n
        self.local = local
        self.target = target
        self.f = f
        self.name = name


examples = {
    1: Example(
        n=2,
        local=Zone(shape='box', low=[-2, -2], up=[2, 2]),
        target=Zone(shape='ball', center=[0, 0], r=1),
        f=[lambda x, u: x[1] + u[0],
           lambda x, u: x[0] - 0.25 * x[0] ** 2 + u[0]],
        name='test'
    ),
    2: Example(
        n=2,
        local=Zone(shape='box', low=[-2, -2], up=[2, 2]),
        target=Zone(shape='box', low=[-1, -1], up=[1, 1]),
        f=[lambda x, u: x[1] + u[0],
           lambda x, u: x[0] - 0.25 * x[0] ** 2 + u[0]],
        name='test1'
    ),
    3: Example(
        n=2,
        local=Zone(shape=Constant.BOX, low=[-2, -2], up=[2, 2]),
        target=Zone(shape=Constant.BOX, low=[-0.5] * 2, up=[0.5] * 2),
        f=[lambda x, u: x[1],
           lambda x, u: (1 - x[0]**2) * x[1] - x[0] + u[0]],
        name='Oscillator'
    ),
    4: Example(
        n=2,
        local=Zone(shape=Constant.BOX, low=[-6, -7*np.pi], up=[6, 7*np.pi/10]),
        target=Zone(shape=Constant.BALL, center=[0, 0], r=1**2),
        f=[lambda x, u: x[1]-x[1]**3/6+x[1]**5/120,
           lambda x, u: -u[0]],
        name='Car'
    )
}


def get_example_by_id(id: int):
    return examples[id]


def get_example_by_name(name: str):
    for ex in examples.values():
        if ex.name == name:
            return ex
    raise ValueError('The example {} was not found.'.format(name))
