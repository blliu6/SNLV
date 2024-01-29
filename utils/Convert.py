from benchmarks.Examplers import Zone, Example
from constants import *
import sympy as sp


def zone_to_constraints(zone: Zone, x):
    if zone.shape == BOX:
        # up - x; low + x
        n = len(zone.low)
        constr = []
        constr.extend([zone.low[i] + x[i], zone.up[i] - x[i]] for i in range(n))
        return constr

    if zone.shape == BALL:
        return []


if __name__ == "__main__":
    pass
