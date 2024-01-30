import torch
import sympy as sp


class Config:
    N_HIDDEN_NEURONS = [10]
    EXAMPLE = None
    ACTIVATION = ['SQUARE']
    BATCH_SIZE = 500
    LEARNING_RATE = 0.01
    LOSS_WEIGHT = (1, 1, 1)
    MARGIN = 0.5
    DEG = [2, 2, 2]
    OPT = torch.optim.AdamW
    LEARNING_LOOPS = 100
    controller = [sp.sympify('0.976262879008059*x1**2 - 1.01104316630542*x1*x2 + 0.101542297866673*x1 + 0.648682174819056*x2**2 - 0.63719262567874*x2')]
    # -0.0133 + 1.2057 * x1 + 1.2299 * x2 + 13.5323 * x1 ** 2 + 81.1611 * x1 * x2 + 57.9627 * x2 ** 2
    # controller = [sp.sympify('1')]
    max_iter = 100

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
