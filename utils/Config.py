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
    DEG = [2, 2, 2, 2]
    OPT = torch.optim.AdamW
    LEARNING_LOOPS = 100
    controller = sp.sympify('x1+x2')

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
