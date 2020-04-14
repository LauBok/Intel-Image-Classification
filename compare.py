import numpy as np

def compare(pred, true):
    return np.mean(pred == true)