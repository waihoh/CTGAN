from ctgan import config as cfg
import argparse

from ctgan.synthesizer import CTGANSynthesizer
import pandas as pd
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description='Command Line Interface')
    parser.add_argument('-opt', '--optimizer', default=None, type=str,
                        metavar='', help='choice of optimizer')
    parser.add_argument('-lr', '--learningrate', default=None, type=float,
                        metavar='', help='learning rate')
    return parser.parse_args()

def parser_func():
    # # this function is a placeholder to update cfg with values from argparse
    # cfg.LEARNING_RATE = 0.1

    # TEST
    args = _parse_args()

    if args.learningrate is not None:
        cfg.LEARNING_RATE = args.learningrate

    if args.optimizer is not None:
        cfg.OPTIMIZER = args.optimizer


if __name__ == '__main__':
    parser_func()

    # using a toy example to test tablegan
    data = pd.DataFrame({
        'continuous1': np.random.random(1000),
        'discrete1': np.repeat([1, 2, 3], [950, 25, 25]),
        'discrete2': np.repeat(["a", "b"], [580, 420]),
        'discrete3': np.repeat([6, 7], [100, 900])
    })

    # index of columns
    discrete_columns = ['discrete1', 'discrete2', 'discrete3']

    # Step 2: Fit CTGAN to your data
    ctgan = CTGANSynthesizer()
    ctgan.fit(data, discrete_columns, epochs=1, trans="Min-Max")

    # 2. Generate synthetic data
    samples_1 = ctgan.sample(10, condition_column='discrete1', condition_value=1)
    print('size of sample_1', samples_1.shape)

