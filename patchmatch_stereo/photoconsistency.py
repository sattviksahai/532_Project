import numpy as np

def SAD(ref1, ref2):
    return np.sum(np.absolute(np.subtract(ref1, ref2)))

# TODO: add code for NCC

