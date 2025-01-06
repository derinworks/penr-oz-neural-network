import json
import numpy as np
import os

class NeuralNetworkModel:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size).tolist()
        self.bias = np.random.randn(output_size).tolist()

    def serialize(self, filepath):
        os.makedirs("models", exist_ok=True)
        full_path = os.path.join("models", filepath)
        model_data = {
            "weights": self.weights,
            "bias": self.bias
        }
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f)

    @classmethod
    def deserialize(cls, filepath):
        full_path = os.path.join("models", filepath)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file {full_path} does not exist.")
        with open(full_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        input_size = len(model_data["weights"])
        output_size = len(model_data["weights"][0])
        model = cls(input_size, output_size)
        model.weights = model_data["weights"]
        model.bias = model_data["bias"]
        return model

    def compute_output(self, activation_vector):
        activation_array = np.array(activation_vector)
        weights_array = np.array(self.weights)
        bias_array = np.array(self.bias)
        output_array = np.dot(activation_array, weights_array) + bias_array
        return output_array.tolist()
