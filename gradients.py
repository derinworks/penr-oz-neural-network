import functions as func
import numpy as np

class Gradients:
    def __init__(self, weights=None, biases=None):
        self.cost_wrt_weights = [np.zeros_like(w) for w in weights] if weights else []
        self.cost_wrt_biases = [np.zeros_like(b) for b in biases] if biases else []

    def __iadd__(self, other):
        for layer in range(len(self.cost_wrt_weights)):
            self.cost_wrt_weights[layer] += other.cost_wrt_weights[layer]
        for layer in range(len(self.cost_wrt_biases)):
            self.cost_wrt_biases[layer] += other.cost_wrt_biases[layer]
        return self

    def divide_by(self, divisor: int):
        """
        Divide gradients by the given integer divisor
        :param divisor: an integer to divide by
        """
        self.cost_wrt_weights = [w / divisor for w in self.cost_wrt_weights]
        self.cost_wrt_biases = [b / divisor for b in self.cost_wrt_biases]

    def compute(self, weights, algos, activations, pre_activations, target):
        """
        Compute gradients
        :param weights: List of NumPy arrays representing weight matrices
        :param algos: List of algorithms to use activation derivative at each layer
        :param activations: List of NumPy arrays representing activation at each layer
        :param pre_activations: List of NumPy arrays representing pre-activation at each layer
        :param target: A NumPy array representing target
        """
        # collect cost gradients
        self.cost_wrt_weights = []
        self.cost_wrt_biases = []
        # stepping backwards in layers (one less than number of activation edges)
        num_layers = len(weights)
        calculator = GradientCalculator(algos[-1], pre_activations[-1], activations[-1], target)
        for layer in reversed(range(num_layers)):
            # with respect to weights
            layer_cost_wrt_weights = calculator.compute_wrt_weights(layer, activations)
            self.cost_wrt_weights.insert(0, layer_cost_wrt_weights)
            # with respect to biases
            layer_cost_wrt_biases = calculator.compute_wrt_biases()
            self.cost_wrt_biases.insert(0, layer_cost_wrt_biases)
            # propagation of derivatives back to previous layer
            calculator.backpropagation(layer, weights, algos, pre_activations)

class GradientCalculator:
    def __init__(self, algo, pre_activation, activation, target):
        # start derivatives at output edge with target
        cost_wrt_output = func.mean_squared_error_derivative(activation, target)
        pre_activation_wrt_activation = self.derivative_with_algo(algo, pre_activation, target)
        self.cost_wrt_activation = cost_wrt_output * pre_activation_wrt_activation

    @classmethod
    def derivative_with_algo(cls, algo, pre_activation, target=None):
        """
        Apply derivative of activation function based on the given algorithm.
        :param algo: The activation algorithm ("sigmoid", "relu", "tanh", "softmax").
        :param pre_activation: Pre-activation NumPy array
        :param target: Target NumPy array (for last layer only)
        :return: Derivative of the activation of output NumPy array.
        """
        if algo == "sigmoid":
            return func.sigmoid_derivative(pre_activation)
        elif algo == "relu":
            return func.relu_derivative(pre_activation)
        elif algo == "tanh":
            return func.tanh_derivative(pre_activation)
        elif algo == "softmax" and target is not None:
            return func.softmax_cross_entropy_gradient(pre_activation, target)
        else:
            raise ValueError(f"Unsupported derivative of activation algorithm: {algo}")

    def backpropagation(self, layer, weights, algos, pre_activations):
        """
        Backpropagation of cost error to previous layers except last one
        :param layer: Number representing current layer
        :param weights: List of NumPy arrays representing weight matrices
        :param algos: List of algorithms to use activation derivative at each layer
        :param pre_activations: List of NumPy arrays representing pre-activation at each layer
        """
        if layer > 0:
            prev_layer = layer - 1
            prev_pre_activation = pre_activations[prev_layer]
            prev_algo = algos[prev_layer]
            prev_pre_activation_wrt_activation = self.derivative_with_algo(prev_algo, prev_pre_activation)
            layer_weights_transposed = weights[layer].T
            prev_layer_weighted_error = np.dot(self.cost_wrt_activation, layer_weights_transposed)
            self.cost_wrt_activation = prev_layer_weighted_error * prev_pre_activation_wrt_activation

    def compute_wrt_weights(self, layer, activations):
        """
        Compute gradient with respect to weights for the current layer
        :param layer: Number representing current layer
        :param activations: List of NumPy arrays representing activation at each layer
        :return: gradient with respect to weights
        """
        activation_wrt_weights = activations[layer]
        return np.outer(activation_wrt_weights, self.cost_wrt_activation)

    def compute_wrt_biases(self):
        """
        Compute gradient with respect to biases for the current layer
        :return: gradient with respect to biases
        """
        # same as with respect to activation
        return self.cost_wrt_activation
