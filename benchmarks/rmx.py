import timeit
import torch
import numpy as np
from utils.Config import Config
from Examplers import get_example_by_name, get_example_by_id
from learn.Cegis import Cegis


def main():
    start = timeit.default_timer()
    activations = ['SKIP']
    hidden_neurons = [10] * len(activations)

    example = get_example_by_name('Oscillator')

    start = timeit.default_timer()
    opts = {
        'ACTIVATION': activations,
        'N_HIDDEN_NEURONS': hidden_neurons,
        "EXAMPLE": example,
        "BATCH_SIZE": 1000,
        'LEARNING_RATE': 0.1,
        'LOSS_WEIGHT': (1, 1, 1),
        'MARGIN': 1,
        "DEG": [2, 2, 2],
        "LEARNING_LOOPS": 100,
        'max_iter': 20
    }
    config = Config(**opts)
    cegis = Cegis(config)
    end = cegis.solve()
    print('Elapsed Time: {}'.format(end - start))


if __name__ == '__main__':
    torch.manual_seed(2024)
    np.random.seed(2024)
    main()
