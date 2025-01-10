import json
import numpy as np
import os
from functions import mean_squared_error, sigmoid, mean_squared_error_derivative, sigmoid_derivative

class NeuralNetworkModel:
    def __init__(self, layer_sizes):
        """
        Initialize a neural network with multiple layers.

        :param layer_sizes: List of integers where each integer represents the size of a layer.
        """
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1]).tolist()
            for i in range(len(layer_sizes) - 1)
        ]
        self.biases = [
            np.random.randn(layer_size).tolist()
            for layer_size in layer_sizes[1:]
        ]

    def serialize(self, filepath):
        os.makedirs("models", exist_ok=True)
        full_path = os.path.join("models", filepath)
        model_data = {
            "weights": self.weights,
            "biases": self.biases
        }
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=4)

    @classmethod
    def deserialize(cls, filepath):
        full_path = os.path.join("models", filepath)
        with open(full_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        layer_sizes = [len(model_data["weights"][0])] + [len(w[0]) for w in model_data["weights"]]
        model = cls(layer_sizes)
        model.weights = model_data["weights"]
        model.biases = model_data["biases"]
        return model

    def compute_output(self, activation_vector, training_vector=None):
        activations = [np.array(activation_vector)]
        output_arrays = []
        for layer_weights, layer_bias in zip(self.weights, self.biases):
            output_arrays.append(np.dot(activations[-1], np.array(layer_weights)) + np.array(layer_bias))
            activations.append(sigmoid(output_arrays[-1]))

        if training_vector is not None:
            training_array = np.array(training_vector)
            cost = mean_squared_error(activations[-1], training_array)
            cost_derivative_wrt_output = mean_squared_error_derivative(activations[-1], training_array)
            output_derivative_wrt_activation = sigmoid_derivative(output_arrays[-1])
            cost_derivative_wrt_activation = cost_derivative_wrt_output * output_derivative_wrt_activation

            cost_derivatives_wrt_weights = []
            cost_derivatives_wrt_biases = []

            for layer in range(len(self.weights) - 1, -1, -1):
                activation_derivative_wrt_weights = activations[layer - 1]
                cost_derivative_wrt_weights = np.outer(activation_derivative_wrt_weights, cost_derivative_wrt_activation)
                cost_derivatives_wrt_weights.insert(0, cost_derivative_wrt_weights.tolist())

                cost_derivative_wrt_biases = cost_derivative_wrt_activation
                cost_derivatives_wrt_biases.insert(0, cost_derivative_wrt_biases.tolist())

                if layer > 0:  # Backpropagate error to previous layers
                    prev_output_derivative_wrt_activation = sigmoid_derivative(output_arrays[layer - 1])
                    prev_layer_weighted_error = np.dot(cost_derivative_wrt_activation, np.array(self.weights[layer]).T)
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

    def train(self, training_data, epochs=100, learning_rate=0.01):
        """
        Train the neural network using the provided training data.

        :param training_data: List of tuples [(activation_vector, training_vector), ...].
        :param epochs: Number of training iterations.
        :param learning_rate: Learning rate for gradient descent.
        """
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            avg_cost_derivatives_wrt_weights = [np.zeros_like(np.array(w)) for w in self.weights]
            avg_cost_derivatives_wrt_biases = [np.zeros_like(np.array(b)) for b in self.biases]
            total_cost = 0

            for activation_vector, training_vector in training_data:
                _, cost, cost_derivatives_wrt_weights, cost_derivatives_wrt_biases = self.compute_output(
                    activation_vector, training_vector
                )
                total_cost += cost
                for layer in range(len(self.weights)):
                    avg_cost_derivatives_wrt_weights[layer] += np.array(cost_derivatives_wrt_weights[layer])
                    avg_cost_derivatives_wrt_biases[layer] += np.array(cost_derivatives_wrt_biases[layer])

            # Average the derivatives
            avg_cost_derivatives_wrt_weights = [w / len(training_data) for w in avg_cost_derivatives_wrt_weights]
            avg_cost_derivatives_wrt_biases = [b / len(training_data) for b in avg_cost_derivatives_wrt_biases]

            # Update weights and biases
            self._train_step(avg_cost_derivatives_wrt_weights, avg_cost_derivatives_wrt_biases, learning_rate)

            # Print progress
            print(f"Epoch {epoch + 1}/{epochs}, Cost: {total_cost / len(training_data):.4f}")
