import torch


class Config:
    N_HIDDEN_NEURONS = [10]
    EXAMPLE = None
    ACTIVATION = ['SKIP']
    batch_size = 500
    LEARNING_RATE = 0.01
    loss_weight = (1, 1, 1)
    SPLIT_D = False
    margin = 0.5
    DEG = [2, 2, 2, 2]
    OPT = torch.optim.AdamW
    R_b = 0.4
    learning_loops = 100

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
