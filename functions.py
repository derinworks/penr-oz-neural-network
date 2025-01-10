import numpy as np

def sigmoid(x):
    """
    Takes a NumPy array x and returns an array of the same shape
    :param x: a NumPy array x
    :return: Activation function: Sigmoid
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Takes a NumPy array x (already sigmoid activated) and returns an array of the same shape
    :param x: a NumPy array x
    :return: Derivative of sigmoid function
    """
    return x * (1 - x)

def mean_squared_error(x1, x2):
    """
    Compares a NumPy array 1 to 2 and calculates cost going from 1 to 2
    :param x1: a NumPy array x1
    :param x2: a NumPy array x2
    :return: the mean squared error between a NumPy array 1 to 2
    """
    return np.sum((x1 - x2) ** 2)

def mean_squared_error_derivative(x1, x2):
    """
    Calculates cost derivative going from 1 to 2
    :param x1: a NumPy array x1
    :param x2: a NumPy array x2
    :return: Derivative of the mean squared error between a NumPy array 1 to 2
    """
    return 2 * (x1 - x2)
