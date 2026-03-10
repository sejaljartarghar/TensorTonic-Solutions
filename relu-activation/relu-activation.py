import numpy as np

def relu(x):
    x = np.array(x, dtype=float)
    return np.maximum(0, x)