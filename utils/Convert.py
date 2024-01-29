from benchmarks.Examplers import Zone, Example
from constants import *
import sympy as sp
import numpy as np


def zone_to_constraints(zone: Zone, x):
    if zone.shape == Constant.BOX:
        # up - x; low + x
        n = len(zone.low)
        constr = []
        for i in range(n):
            constr.extend([zone.low[i] + x[i], zone.up[i] - x[i]])
        return np.array(constr)

    if zone.shape == Constant.BALL:
        # r - \sum (x_i - center_i)**2
        return np.array([zone.r - sum((x[i] - zone.center[i]) ** 2 for i in range(len(zone.center)))])


if __name__ == "__main__":
    example = Example(
        n=2,
        local=Zone(shape='box', low=[-2, -2], up=[2, 2]),
        target=Zone(shape='ball', center=[0, 0], r=1),
        f=[lambda x, u: x[1] + u[0],
           lambda x, u: x[0] - 0.25 * x[0] ** 2 + u[0]],
        name='test'
    )
    x = sp.symbols([f'x{i + 1}' for i in range(example.n)])
    print(zone_to_constraints(example.local, x))
    print(zone_to_constraints(example.target, x))
