import json
import logging
import os
import numpy as np
import functions as func
from adam_optimizer import AdamOptimizer
from gradients import Gradients
import time
from datetime import datetime as dt

log = logging.getLogger(__name__)

class NeuralNetworkModel:
    def __init__(self, model_id, layer_sizes, weight_algo="xavier", bias_algo="random", activation_algos=None):
        """
        Initialize a neural network with multiple layers.
        :param layer_sizes: List of integers where each integer represents the size of a layer.
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for biases (default: "random")
        :param activation_algos: Activation algorithms (default: "sigmoid")
        """
        self.model_id = model_id

        scaling_factors = {
            "xavier": lambda i: np.sqrt(1 / layer_sizes[i]),
            "he": lambda i: np.sqrt(2 / layer_sizes[i]),
            "gaussian": lambda i: 1
        }
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1])
             * scaling_factors.get(weight_algo, scaling_factors["gaussian"])(i)
            for i in range(len(layer_sizes) - 1)
        ]
        self.weight_optimizer = AdamOptimizer()

        self.biases = [
            np.zeros(layer_size) if bias_algo == "zeros"
            else np.random.randn(layer_size)
            for layer_size in layer_sizes[1:]
        ]
        self.bias_optimizer = AdamOptimizer()

        self.activation_algos = activation_algos or ["sigmoid"] * (len(layer_sizes) - 1)

        self.progress = []

        # Training data buffer
        self.training_data_buffer = []

        # Calculate training buffer and sample size based on total parameters
        self.training_buffer_size = self._calculate_buffer_size(layer_sizes)
        self.training_sample_size = int(self.training_buffer_size * 0.1) # sample 10% of buffer

    @staticmethod
    def _calculate_buffer_size(layer_sizes):
        """
        Calculate training data buffer size based on total number of parameters in the network.
        """
        total_params = sum(
            layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )
        return total_params  # Buffer size is equal to total parameters

    @classmethod
    def activate_with_algo(cls, algo, pre_activation):
        """
        Apply activation function based on the given algorithm.
        :param algo: The activation algorithm ("sigmoid", "relu", "tanh", softmax).
        :param pre_activation: Pre-activated NumPy array.
        :return: Activated NumPy array.
        """
        if algo == "sigmoid":
            return func.sigmoid(pre_activation)
        elif algo == "relu":
            return func.relu(pre_activation)
        elif algo == "tanh":
            return func.tanh(pre_activation)
        elif algo == "softmax":
            return func.softmax(pre_activation)
        else:
            raise ValueError(f"Unsupported activation algorithm: {algo}")

    def get_model_data(self):
        return {
            "weights": [w.tolist() for w in self.weights],
            "weight_optimizer_state": self.weight_optimizer.state,
            "biases": [b.tolist() for b in self.biases],
            "bias_optimizer_state": self.bias_optimizer.state,
            "activation_algos": self.activation_algos,
            "progress": self.progress,
            "training_data_buffer": self.training_data_buffer,
        }

    def set_model_data(self, model_data):
        self.weights = [np.array(w) for w in model_data["weights"]]
        self.weight_optimizer.state = model_data["weight_optimizer_state"]
        self.biases = [np.array(b) for b in model_data["biases"]]
        self.bias_optimizer.state = model_data["bias_optimizer_state"]
        self.activation_algos = model_data["activation_algos"]
        self.progress = model_data["progress"]
        self.training_data_buffer = model_data["training_data_buffer"]

    def serialize(self):
        filepath = f"model_{self.model_id}.json"
        os.makedirs("models", exist_ok=True)
        full_path = os.path.join("models", filepath)
        model_data = self.get_model_data()
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
        model.set_model_data(model_data)
        return model

    @classmethod
    def delete(cls, model_id):
        filepath = f"model_{model_id}.json"
        full_path = os.path.join("models", filepath)
        try:
            os.remove(full_path)
        except FileNotFoundError as e:
            log.warning(f"Failed to delete: {str(e)}")

    def compute_output(self, activation_vector, target_vector=None, dropout_rate=0.0):
        """
        Compute activated output and optionally also cost compared to the provided training data.
        :param activation_vector: Activation vector
        :param target_vector: Target vector (optional)
        :param dropout_rate: Fraction of neurons to drop during training for hidden layers (optional)
        """
        activations = [np.array(activation_vector)]
        pre_activations = []
        num_layers = len(self.weights)
        for layer in range(num_layers):
            algo = self.activation_algos[layer]
            pre_activation = np.dot(activations[-1], self.weights[layer]) + self.biases[layer]
            if algo == "relu" and layer < num_layers - 1:
                # stabilize output in hidden layers prevent overflow with ReLU activations
                pre_activation = func.batch_norm(pre_activation)
            pre_activations.append(pre_activation)
            activation = self.activate_with_algo(algo, pre_activations[-1])
            if layer < num_layers - 1:  # Hidden layers only
                # Apply dropout only to hidden layers
                activation = func.apply_dropout(activation, dropout_rate)
            activations.append(activation)

        cost = None
        gradients = Gradients()
        if target_vector is not None:
            target = np.array(target_vector)
            cost = func.mean_squared_error(activations[-1], target)
            gradients.compute(self.weights, self.activation_algos, activations, pre_activations, target)

        return activations[-1].tolist(), cost, gradients

    def _train_step(self, avg_gradients, learning_rate, l2_lambda):
        """
        Update the weights and biases of the neural network using the averaged cost derivatives.
        :param avg_gradients: Averaged gradients holding lists of NumPy arrays for weights and biases
        :param learning_rate: Learning rate for gradient descent.
        :param l2_lambda: L2 regularization strength.
        """
        # Optimize weight gradients
        optimized_weight_steps = self.weight_optimizer.step(avg_gradients.cost_wrt_weights, learning_rate)
        # Update weights by optimized gradients
        for layer in range(len(self.weights)):
            self.weights[layer] -= optimized_weight_steps[layer]
        # Update weights with L2 regularization
        for layer in range(len(self.weights)):
            l2_penalties = l2_lambda * self.weights[layer]
            self.weights[layer] -= l2_penalties
        # Optimize bias gradients
        optimized_bias_steps = self.bias_optimizer.step(avg_gradients.cost_wrt_biases, learning_rate)
        # Update biases
        for layer in range(len(self.biases)):
            self.biases[layer] -= optimized_bias_steps[layer]

    def train(self, training_data, epochs=100, learning_rate=0.01, decay_rate=0.9, dropout_rate=0.2,
              l2_lambda=0.001):
        """
        Train the neural network using the provided training data.

        :param training_data: List of tuples [(activation_vector, target_vector), ...].
        :param epochs: Number of training iterations.
        :param learning_rate: Learning rate for gradient descent.
        :param decay_rate: Decay rate of learning rate for finer gradient descent
        :param dropout_rate: Fraction of neurons to drop during training for hidden layers
        :param l2_lambda: L2 regularization strength
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
            training_data_sample = training_data[:self.training_sample_size]

            # Calculate total cost and average gradients
            avg_gradients = Gradients(self.weights, self.biases)
            total_cost = 0
            for activation_vector, target_vector in training_data_sample:
                _, cost, gradients = self.compute_output(activation_vector, target_vector, dropout_rate)
                total_cost += cost
                avg_gradients += gradients # sum up gradients
            # take average of by size of sample
            avg_gradients.divide_by(len(training_data_sample))

            # Update weights and biases
            current_learning_rate = learning_rate * (decay_rate ** epoch)
            self._train_step(avg_gradients, current_learning_rate, l2_lambda)

            # Record progress
            self.progress.append({
                "dt": dt.now().isoformat(),
                "epoch": epoch + 1,
                "cost": total_cost / len(training_data_sample)
            })
            last_progress = self.progress[-1]
            print(f"Model {self.model_id}: {last_progress["dt"]} - Epoch {last_progress["epoch"]}, "
                  f"Cost: {last_progress["cost"]:.4f} ")

            # Serialize model after 10 secs while training
            if time.time() - last_serialized >= 10:
                self.serialize()

        # Serialize model after training
        self.serialize()
