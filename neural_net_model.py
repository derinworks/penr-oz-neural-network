import json
import logging
import os
import numpy as np
import functions as func
import time
from datetime import datetime as dt

log = logging.getLogger(__name__)

class NeuralNetworkModel:
    def __init__(self, model_id, layer_sizes, init_algo="xavier"):
        """
        Initialize a neural network with multiple layers.

        :param layer_sizes: List of integers where each integer represents the size of a layer.
        :param init_algo: Initialization algorithm for weights (default: "xavier").
        """
        self.model_id = model_id

        scaling_factors = {
            "xavier": lambda i: np.sqrt(1 / layer_sizes[i]),
            "he": lambda i: np.sqrt(2 / layer_sizes[i]),
            "gaussian": lambda i: 1
        }
        self.weights = [
            (np.random.randn(layer_sizes[i], layer_sizes[i + 1])
             * scaling_factors.get(init_algo, scaling_factors["gaussian"])(i)).tolist()
            for i in range(len(layer_sizes) - 1)
        ]

        self.biases = [
            np.random.randn(layer_size).tolist()
            for layer_size in layer_sizes[1:]
        ]
        self.progress = []

        # Training data buffer
        self.training_data_buffer = []

        # Calculate training buffer size based on total parameters
        self.training_buffer_size = self._calculate_buffer_size(layer_sizes)

    @staticmethod
    def _calculate_buffer_size(layer_sizes):
        """
        Calculate training data buffer size based on total number of parameters in the network.
        """
        total_params = sum(
            layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )
        return int(0.1 * total_params)  # Buffer size is 10% of total parameters

    @classmethod
    def activate_with_algo(cls, algo, output_array):
        """
        Apply activation function based on the given algorithm.
        :param algo: The activation algorithm ("sigmoid", "relu", "tanh").
        :param output_array: Output NumPy array.
        :return: Activated NumPy array.
        """
        if algo == "sigmoid":
            return func.sigmoid(output_array)
        elif algo == "relu":
            return func.relu(output_array)
        elif algo == "tanh":
            return func.tanh(output_array)
        else:
            raise ValueError(f"Unsupported activation algorithm: {algo}")

    @classmethod
    def derivative_with_algo(cls, algo, output_array):
        """
        Apply derivative of activation function based on the given algorithm.
        :param algo: The activation algorithm ("sigmoid", "relu", "tanh").
        :param output_array: Output NumPy array
        :return: Derivative of the activation of output NumPy array.
        """
        if algo == "sigmoid":
            return func.sigmoid_derivative(output_array)
        elif algo == "relu":
            return func.relu_derivative(output_array)
        elif algo == "tanh":
            return func.tanh_derivative(output_array)
        else:
            raise ValueError(f"Unsupported derivative of activation algorithm: {algo}")

    def serialize(self):
        filepath = f"model_{self.model_id}.json"
        os.makedirs("models", exist_ok=True)
        full_path = os.path.join("models", filepath)
        model_data = {
            "weights": self.weights,
            "biases": self.biases,
            "progress": self.progress,
            "training_buffer_size": self.training_buffer_size,
            "training_data_buffer": self.training_data_buffer
        }
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=4)
        log.info(f"Model saved successfully: {full_path}")

    @classmethod
    def deserialize(cls, model_id):
        filepath = f"model_{model_id}.json"
        full_path = os.path.join("models", filepath)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
        except FileNotFoundError as e:
            log.error(f"File not found error occurred: {str(e)}")
            raise KeyError(f"Model {model_id} not created yet.")
        layer_sizes = [len(model_data["weights"][0])] + [len(w[0]) for w in model_data["weights"]]
        model = cls(model_id, layer_sizes)
        model.weights = model_data["weights"]
        model.biases = model_data["biases"]
        model.progress = model_data["progress"]
        model.training_buffer_size = model_data["training_buffer_size"]
        model.training_data_buffer = model_data["training_data_buffer"]
        return model

    @classmethod
    def delete(cls, model_id):
        filepath = f"model_{model_id}.json"
        full_path = os.path.join("models", filepath)
        try:
            os.remove(full_path)
        except FileNotFoundError as e:
            log.warning(f"Failed to delete: {str(e)}")

    def compute_output(self, activation_vector, activation_algo="sigmoid", training_vector=None):
        """
        Compute activated output and optionally also cost compared to the provided training data.

        :param activation_vector: Activation vector
        :param activation_algo: Algorithm used to activate
        :param training_vector: Training vector (optional)
        """
        activations = [np.array(activation_vector)]
        output_arrays = []
        for layer_weights, layer_bias in zip(self.weights, self.biases):
            output_arrays.append(np.dot(activations[-1], np.array(layer_weights)) + np.array(layer_bias))
            activations.append(self.activate_with_algo(activation_algo, output_arrays[-1]))

        if training_vector is not None:
            training_array = np.array(training_vector)
            cost = func.mean_squared_error(activations[-1], training_array)
            # collect cost derivatives
            cost_derivatives_wrt_weights = []
            cost_derivatives_wrt_biases = []
            # start derivatives at output edge with training array
            cost_derivative_wrt_output = func.mean_squared_error_derivative(activations[-1], training_array)
            output_derivative_wrt_activation = self.derivative_with_algo(activation_algo, output_arrays[-1])
            cost_derivative_wrt_activation = cost_derivative_wrt_output * output_derivative_wrt_activation
            # now stepping backwards in layers (one less than number of activation edges)
            for layer in range(len(self.weights) - 1, -1, -1):
                # one more activation edge than layers so current edge index = (layer index + 1), previous = current - 1
                # which means previous edge index = (layer + 1) - 1 = current layer index
                prev_activation_edge = layer
                # with respect to weights
                prev_activations = activations[prev_activation_edge] # previous edge of current layer
                activation_derivative_wrt_weights = prev_activations # cost derivative is the previous activations
                cost_derivative_wrt_weights = np.outer(activation_derivative_wrt_weights, cost_derivative_wrt_activation)
                cost_derivatives_wrt_weights.insert(0, cost_derivative_wrt_weights.tolist())
                # with respect to biases
                # cost derivative is same as current edge derivative of activations with respect to weights
                cost_derivative_wrt_biases = cost_derivative_wrt_activation
                cost_derivatives_wrt_biases.insert(0, cost_derivative_wrt_biases.tolist())
                # Backpropagation of cost error to previous layers except last one
                if layer > 0:
                    # same number of output edges as layers
                    # each layer has output on current edge. previous output edge = current layer - 1
                    prev_output_edge = layer - 1
                    prev_output_array = output_arrays[prev_output_edge]
                    prev_output_derivative_wrt_activation = self.derivative_with_algo(activation_algo, prev_output_array)
                    layer_weights_array_transposed = np.array(self.weights[layer]).T
                    prev_layer_weighted_error = np.dot(cost_derivative_wrt_activation, layer_weights_array_transposed)
                    cost_derivative_wrt_activation = prev_layer_weighted_error * prev_output_derivative_wrt_activation
        else:
            cost = None
            cost_derivatives_wrt_weights = None
            cost_derivatives_wrt_biases = None

        return activations[-1].tolist(), cost, cost_derivatives_wrt_weights, cost_derivatives_wrt_biases

    def _train_step(self, avg_cost_derivatives_wrt_weights, avg_cost_derivatives_wrt_biases, learning_rate):
        """
        Update the weights and biases of the neural network using the averaged cost derivatives.

        :param avg_cost_derivatives_wrt_weights: List of averaged cost derivatives with respect to weights.
        :param avg_cost_derivatives_wrt_biases: List of averaged cost derivatives with respect to biases.
        :param learning_rate: Learning rate for gradient descent.
        """
        # Update weights
        for layer in range(len(self.weights)):
            self.weights[layer] = (np.array(self.weights[layer]) -
                                   learning_rate * np.array(avg_cost_derivatives_wrt_weights[layer])).tolist()
        # Update biases
        for layer in range(len(self.biases)):
            self.biases[layer] = (np.array(self.biases[layer]) -
                                  learning_rate * np.array(avg_cost_derivatives_wrt_biases[layer])).tolist()

    def train(self, training_data, activation_algo='sigmoid', epochs=100, learning_rate=0.01, decay_rate=0.9):
        """
        Train the neural network using the provided training data.

        :param training_data: List of tuples [(activation_vector, training_vector), ...].
        :param activation_algo: Algorithm used to activate
        :param epochs: Number of training iterations.
        :param learning_rate: Learning rate for gradient descent.
        :param decay_rate: Decay rate of learning rate for finer gradient descent
        """
        # Combine incoming training data with buffered data
        self.training_data_buffer.extend(training_data)

        # Check if buffer size is sufficient
        if len(self.training_data_buffer) < self.training_buffer_size:
            print(f"Model {self.model_id}: Insufficient training data. "
                  f"Current buffer size: {len(self.training_data_buffer)}, "
                  f"required: {self.training_buffer_size}")
            self.serialize() # serialize model with partial training data for next time
            return

        # Proceed with training using combined data if buffer size is sufficient
        training_data = self.training_data_buffer
        self.training_data_buffer = []  # Clear buffer

        self.progress = []
        last_serialized = time.time()
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            avg_cost_derivatives_wrt_weights = [np.zeros_like(np.array(w)) for w in self.weights]
            avg_cost_derivatives_wrt_biases = [np.zeros_like(np.array(b)) for b in self.biases]
            total_cost = 0

            for activation_vector, training_vector in training_data:
                _, cost, cost_derivatives_wrt_weights, cost_derivatives_wrt_biases = self.compute_output(
                    activation_vector, activation_algo, training_vector
                )
                total_cost += cost
                for layer in range(len(self.weights)):
                    avg_cost_derivatives_wrt_weights[layer] += np.array(cost_derivatives_wrt_weights[layer])
                    avg_cost_derivatives_wrt_biases[layer] += np.array(cost_derivatives_wrt_biases[layer])

            # Average the derivatives
            avg_cost_derivatives_wrt_weights = [w / len(training_data) for w in avg_cost_derivatives_wrt_weights]
            avg_cost_derivatives_wrt_biases = [b / len(training_data) for b in avg_cost_derivatives_wrt_biases]

            # Update weights and biases
            current_learning_rate = learning_rate * (decay_rate ** epoch)
            self._train_step(avg_cost_derivatives_wrt_weights, avg_cost_derivatives_wrt_biases, current_learning_rate)

            # Record progress
            self.progress.append(f"{dt.now().isoformat()} - Epoch {epoch + 1}/{epochs}, Cost: {total_cost / len(training_data):.4f}")
            print(f"Model {self.model_id}: {self.progress[-1]}")

            # Serialize model after 10 secs while training
            if time.time() - last_serialized >= 10:
                self.serialize()

        # Serialize model after training
        self.serialize()
