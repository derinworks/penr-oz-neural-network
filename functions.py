import numpy as np

def sigmoid(x):
    """
    Activation function: Sigmoid
    Takes a NumPy array x and returns an array of the same shape
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of sigmoid function
    Takes a NumPy array x (already sigmoid activated) and returns an array of the same shape
    """
    return x * (1 - x)
