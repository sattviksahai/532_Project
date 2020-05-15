import numpy as np

def SAD(ref1, ref2):
    return np.sum(np.absolute(np.subtract(ref1, ref2)))

# TODO: add code for NCC

def NCC(ref1, ref2):
    return np.sum(np.matmul(ref1 - np.mean(ref1), ref2 - np.mean(ref2)))/np.sqrt(np.sum(np.square(ref1 - np.mean(ref1)))*np.sum(np.square(ref2 - np.mean(ref2))))
