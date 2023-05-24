import os

import numpy as np

import torch


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # Hyper parameters.
    learning_rate = 0.0002
    epochs = 20
    batch_size = 64

    checkpoints_path = './checkpoints'
    os.makedirs(checkpoints_path, exist_ok=True)

    # Data load.

    # Define model.
    model = None

if __name__ == '__main__':
    main()