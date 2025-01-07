import json
import numpy as np
import os
from functions import mean_squared_error

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
        for weights, bias in zip(self.weights, self.biases):
            weights_array = np.array(weights)
            bias_array = np.array(bias)
            activations.append(np.dot(activations[-1], weights_array) + bias_array)

        output_array = activations[-1]

        if training_vector is not None:
            training_array = np.array(training_vector)
            cost = mean_squared_error(output_array, training_array)
        else:
            cost = None

        return output_array.tolist(), cost
