import numpy as np

def sigmoid(x):
    """
    Takes a NumPy array x and returns an array of the same shape, avoiding overflow errors.
    :param x: a NumPy array x
    :return: Activation function: Sigmoid
    """
    x = np.clip(x, -500, 500)  # Avoid overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Takes a NumPy array x (raw pre-activation or sigmoid activated) and returns the derivative of sigmoid.
    :param x: a NumPy array x
    :return: Derivative of sigmoid function
    """
    sig = sigmoid(x)  # Ensure sigmoid is applied if x is pre-activation
    return sig * (1 - sig)

def relu(x):
    """
    Takes a NumPy array x and applies the ReLU activation function.
    :param x: a NumPy array x
    :return: ReLU activation of x
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Takes a NumPy array x and returns the derivative of the ReLU function.
    :param x: a NumPy array x
    :return: Derivative of ReLU activation
    """
    return np.where(x > 0, 1, 0)

def tanh(x):
    """
    Takes a NumPy array x and applies the tanh activation function.
    :param x: a NumPy array x
    :return: tanh activation of x
    """
    return np.tanh(x)

def tanh_derivative(x):
    """
    Takes a NumPy array x and returns the derivative of the tanh function.
    :param x: a NumPy array x
    :return: Derivative of tanh activation
    """
    return 1 - np.tanh(x) ** 2

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

