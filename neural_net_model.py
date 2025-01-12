import json
import os
from functions import *

class NeuralNetworkModel:
    def __init__(self, model_id, layer_sizes, init_algo="xavier"):
        """
        Initialize a neural network with multiple layers.

        :param layer_sizes: List of integers where each integer represents the size of a layer.
        :param init_algo: Initialization algorithm for weights (default: "xavier").
        """
        self.model_id = model_id
        if init_algo == "xavier":
            self.weights = [
                (np.random.randn(layer_sizes[i], layer_sizes[i + 1])
                 * np.sqrt(1 / layer_sizes[i])).tolist()
                for i in range(len(layer_sizes) - 1)
            ]
        elif init_algo == "he":
            self.weights = [
                (np.random.randn(layer_sizes[i], layer_sizes[i + 1])
                 * np.sqrt(2 / layer_sizes[i])).tolist()
                for i in range(len(layer_sizes) - 1)
            ]
        else: # init algo gaussian
            self.weights = [
                np.random.randn(layer_sizes[i], layer_sizes[i + 1]).tolist()
                for i in range(len(layer_sizes) - 1)
            ]

        self.biases = [
            np.random.randn(layer_size).tolist()
            for layer_size in layer_sizes[1:]
        ]
        self.progress = []

    @classmethod
    def activate_with_algo(cls, algo, output_array):
        """
        Apply activation function based on the given algorithm.
        :param algo: The activation algorithm ("sigmoid", "relu", "tanh").
        :param output_array: Output NumPy array.
        :return: Activated NumPy array.
        """
        if algo == "sigmoid":
            return sigmoid(output_array)
        elif algo == "relu":
            return relu(output_array)
        elif algo == "tanh":
            return tanh(output_array)
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
            return sigmoid_derivative(output_array)
        elif algo == "relu":
            return relu_derivative(output_array)
        elif algo == "tanh":
            return tanh_derivative(output_array)
        else:
            raise ValueError(f"Unsupported derivative of activation algorithm: {algo}")

    def serialize(self):
        filepath = f"model_{self.model_id}.json"
        os.makedirs("models", exist_ok=True)
        full_path = os.path.join("models", filepath)
        model_data = {
            "weights": self.weights,
            "biases": self.biases,
            "progress": self.progress
        }
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=4)
        print(f"Model saved successfully: {full_path}")

    @classmethod
    def deserialize(cls, model_id):
        filepath = f"model_{model_id}.json"
        full_path = os.path.join("models", filepath)
        with open(full_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        layer_sizes = [len(model_data["weights"][0])] + [len(w[0]) for w in model_data["weights"]]
        model = cls(model_id, layer_sizes)
        model.weights = model_data["weights"]
        model.biases = model_data["biases"]
        model.progress = model_data["progress"]
        return model

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
            cost = mean_squared_error(activations[-1], training_array)
            cost_derivative_wrt_output = mean_squared_error_derivative(activations[-1], training_array)
            output_derivative_wrt_activation = self.derivative_with_algo(activation_algo, output_arrays[-1])
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
                    prev_output_derivative_wrt_activation = self.derivative_with_algo(activation_algo, output_arrays[layer - 1])
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

    def train(self, training_data, activation_algo='sigmoid', epochs=100, learning_rate=0.01):
        """
        Train the neural network using the provided training data.

        :param training_data: List of tuples [(activation_vector, training_vector), ...].
        :param activation_algo: Algorithm used to activate
        :param epochs: Number of training iterations.
        :param learning_rate: Learning rate for gradient descent.
        """
        self.progress = []
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
            self._train_step(avg_cost_derivatives_wrt_weights, avg_cost_derivatives_wrt_biases, learning_rate)

            # Record progress
            self.progress.append(f"Epoch {epoch + 1}/{epochs}, Cost: {total_cost / len(training_data):.4f}")
            print(self.progress[-1])

        # Serialize model after training
        self.serialize()
