import numpy as np

def sigmoid(x):
    """
    Takes a NumPy array x and returns an array of the same shape, avoiding overflow errors.
    :param x: a NumPy array x
    :return: Sigmoid activation of x
    """
    x = np.clip(x, -500, 500)  # Avoid overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Takes a NumPy array x (activated) and returns the derivative of sigmoid.
    :param x: a NumPy array x
    :return: Derivative of sigmoid activation
    """
    return x * (1 - x)

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
    # Check for invalid values
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or Inf values.")
    return np.tanh(x)

def tanh_derivative(x):
    """
    Takes a NumPy array x (activated) and returns the derivative of the tanh function.
    :param x: a NumPy array x (activated)
    :return: Derivative of tanh activation
    """
    return 1 - x ** 2

def mean_squared_error(x1, x2):
    """
    Compares a NumPy array 1 to 2 and calculates cost going from 1 to 2
    :param x1: a NumPy array x1
    :param x2: a NumPy array x2
    :return: the mean squared error between a NumPy array 1 to 2
    """
    return np.sum((x2 - x1) ** 2) / len(x1)

def mean_squared_error_derivative(x1, x2):
    """
    Calculates cost derivative going from 1 to 2
    :param x1: a NumPy array x1
    :param x2: a NumPy array x2
    :return: Derivative of the mean squared error between a NumPy array 1 to 2
    """
    return 2 * (x2 - x1)

def softmax(x):
    """
    Compute the softmax of a NumPy array.
    :param x: a NumPy array x
    :return: Softmax probabilities of the input.
    """
    # Check for invalid values
    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or Inf values.")
    # Shift input values for numerical stability (prevent overflow/underflow)
    shift_x = x - np.max(x)
    # Exponential of shifted values
    exp_x = np.exp(shift_x)
    # Normalize by the sum of exponential
    return exp_x / np.sum(exp_x)


def cross_entropy_gradient(x, y):
    """
    Compute the gradient of the cross-entropy loss with softmax for a single sample.
    :param x: a NumPy array x (1D softmax activated probabilities for a single sample).
    :param y: a NumPy array y (1D expected output, a probability distribution).
    :return: Gradient of the loss with respect to the softmax activation.
    """
    return y - x

def cross_entropy_loss(x, y):
    """
    Compute the cross-entropy loss for a single sample.
    :param x: an input NumPy array (1D pre-activation logits for a single sample).
    :param y: an expected output NumPy array (1D expected output, either one-hot encoded or a probability distribution).
    :return: Cross entropy loss.
    """
    # Clipping to avoid log(0) issues
    eps = 1e-12
    x = np.array([max(min(p, 1 - eps), eps) for p in x])
    # Compute cross-entropy loss
    return -np.sum(y * np.log(x))

def batch_norm(x, epsilon=1e-5):
    """
    Applies batch normalization to given NumPy array
    :param x: a NumPy array
    :param epsilon: normalization option
    :return:
    """
    mean = np.mean(x, axis=0)
    variance = np.var(x, axis=0)
    return (x - mean) / np.sqrt(variance + epsilon)

def apply_dropout(x, dropout_rate):
    """
    Applies dropout to the given vector.
    :param x: a NumPy array
    :param dropout_rate: The fraction of vector entries to drop (e.g., 0.2 for 20%).
    :return: a NumPy array with dropout applied.
    """
    if dropout_rate <= 0.0:
        return x  # No drop out if rate is invalid
    elif dropout_rate >= 1.0:
        return np.zeros_like(x) # All drops out on probabilistic certainty
    dropout_mask = np.random.rand(*x.shape) > dropout_rate
    return x * dropout_mask / (1.0 - dropout_rate)
