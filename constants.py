import numpy as np

SOURCE = np.array([
    [450, 20],
    [625, 20],
    [820, 700],
    [5, 700]
])

TARGET_WIDTH = 10
TARGET_HEIGHT = 40

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])