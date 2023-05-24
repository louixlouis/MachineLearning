import numpy as np

from model import *

if __name__=='__main__':
    # Load dataset.

    # What is K?
    K = None
    hwf = None
    height, width, focal = hwf
    height, width = int(height), int(width)

    if K is None:
        K = np.array([
            [focal, 0, 0.5*width],
            [0, focal, 0.5*height],
            [0, 0, 1]
        ])

    # Create directories.

    # Create model.
    model = None