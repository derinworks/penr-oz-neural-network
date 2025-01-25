import json
import logging
import os
import numpy as np
import functions as func
from adam_optimizer import AdamOptimizer
import time
from datetime import datetime as dt

log = logging.getLogger(__name__)

class NeuralNetworkModel:
    def __init__(self, model_id, layer_sizes, weight_algo="xavier", bias_algo="random"):
        """
        Initialize a neural network with multiple layers.
        :param layer_sizes: List of integers where each integer represents the size of a layer.
        :param weight_algo: Initialization algorithm for weights (default: "xavier").
        :param bias_algo: Initialization algorithm for biases (default: "random")
        """
        self.model_id = model_id

        scaling_factors = {
            "xavier": lambda i: np.sqrt(1 / layer_sizes[i]),
            "he": lambda i: np.sqrt(2 / layer_sizes[i]),
            "gaussian": lambda i: 1
        }
        self.weights = [
            (np.random.randn(layer_sizes[i], layer_sizes[i + 1])
             * scaling_factors.get(weight_algo, scaling_factors["gaussian"])(i)).tolist()
            for i in range(len(layer_sizes) - 1)
        ]
        self.weight_optimizer = AdamOptimizer()

        self.biases = [
            (np.zeros(layer_size) if bias_algo == "zeros"
             else np.random.randn(layer_size)).tolist()
            for layer_size in layer_sizes[1:]
        ]
        self.bias_optimizer = AdamOptimizer()

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
    def activate_with_algo(cls, algo, pre_activation_array):
        """
        Apply activation function based on the given algorithm.
        :param algo: The activation algorithm ("sigmoid", "relu", "tanh", softmax).
        :param pre_activation_array: Pre-activated NumPy array.
        :return: Activated NumPy array.
        """
        if algo == "sigmoid":
            return func.sigmoid(pre_activation_array)
        elif algo == "relu":
            return func.relu(pre_activation_array)
        elif algo == "tanh":
            return func.tanh(pre_activation_array)
        elif algo == "softmax":
            return func.softmax(pre_activation_array)
        else:
            raise ValueError(f"Unsupported activation algorithm: {algo}")

    @classmethod
    def derivative_with_algo(cls, algo, pre_activation_array, target_array=None):
        """
        Apply derivative of activation function based on the given algorithm.
        :param algo: The activation algorithm ("sigmoid", "relu", "tanh", "softmax").
        :param pre_activation_array: Pre-activation NumPy array
        :param target_array: Target NumPy array (for last layer only)
        :return: Derivative of the activation of output NumPy array.
        """
        if algo == "sigmoid":
            return func.sigmoid_derivative(pre_activation_array)
        elif algo == "relu":
            return func.relu_derivative(pre_activation_array)
        elif algo == "tanh":
            return func.tanh_derivative(pre_activation_array)
        elif algo == "softmax" and target_array is not None:
            return func.softmax_cross_entropy_gradient(pre_activation_array, target_array)
        else:
            raise ValueError(f"Unsupported derivative of activation algorithm: {algo}")

    def serialize(self):
        filepath = f"model_{self.model_id}.json"
        os.makedirs("models", exist_ok=True)
        full_path = os.path.join("models", filepath)
        model_data = {
            "weights": self.weights,
            "weight_optimizer_state": self.weight_optimizer.state,
            "biases": self.biases,
            "bias_optimizer_state": self.bias_optimizer.state,
            "progress": self.progress,
            "training_data_buffer": self.training_data_buffer,
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
        model.weight_optimizer.state = model_data["weight_optimizer_state"]
        model.biases = model_data["biases"]
        model.bias_optimizer.state = model_data["bias_optimizer_state"]
        model.progress = model_data["progress"]
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

    def compute_output(self, activation_vector, activation_algos, target_vector=None, dropout_rate=0.0):
        """
        Compute activated output and optionally also cost compared to the provided training data.
        :param activation_vector: Activation vector
        :param activation_algos: Algorithms used to activate
        :param target_vector: Target vector (optional)
        :param dropout_rate: Fraction of neurons to drop during training for hidden layers (optional)
        """
        activations = [np.array(activation_vector)]
        pre_activation_arrays = []
        num_layers = len(self.weights)
        for layer in range(num_layers):
            layer_weights = self.weights[layer]
            layer_bias = self.biases[layer]
            algo = activation_algos[layer]
            pre_activation_array = np.dot(activations[-1], np.array(layer_weights)) + np.array(layer_bias)
            if algo == "relu" and layer < num_layers - 1:
                # stabilize output in hidden layers prevent overflow with ReLU activations
                pre_activation_array = func.batch_norm(pre_activation_array)
            pre_activation_arrays.append(pre_activation_array)
            activation_array = self.activate_with_algo(algo, pre_activation_arrays[-1])
            if layer < num_layers - 1:  # Hidden layers only
                # Apply dropout only to hidden layers
                activation_array = func.apply_dropout(activation_array, dropout_rate)
            activations.append(activation_array)

        if target_vector is not None:
            target_array = np.array(target_vector)
            cost = func.mean_squared_error(activations[-1], target_array)
            # collect cost derivatives
            cost_derivatives_wrt_weights = []
            cost_derivatives_wrt_biases = []
            # start derivatives at output edge with training array
            cost_derivative_wrt_output = func.mean_squared_error_derivative(activations[-1], target_array)
            pre_activation_derivative_wrt_activation = self.derivative_with_algo(activation_algos[-1], pre_activation_arrays[-1], target_array)
            cost_derivative_wrt_activation = cost_derivative_wrt_output * pre_activation_derivative_wrt_activation
            # now stepping backwards in layers (one less than number of activation edges)
            for layer in reversed(range(num_layers)):
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
                    prev_layer = layer - 1
                    prev_pre_activation_array = pre_activation_arrays[prev_layer]
                    prev_algo = activation_algos[prev_layer]
                    prev_pre_activation_derivative_wrt_activation = self.derivative_with_algo(prev_algo, prev_pre_activation_array)
                    layer_weights_array_transposed = np.array(self.weights[layer]).T
                    prev_layer_weighted_error = np.dot(cost_derivative_wrt_activation, layer_weights_array_transposed)
                    cost_derivative_wrt_activation = prev_layer_weighted_error * prev_pre_activation_derivative_wrt_activation
        else:
            cost = None
            cost_derivatives_wrt_weights = None
            cost_derivatives_wrt_biases = None

        return activations[-1].tolist(), cost, cost_derivatives_wrt_weights, cost_derivatives_wrt_biases

    def _train_step(self, avg_cost_derivatives_wrt_weights, avg_cost_derivatives_wrt_biases, learning_rate):
        """
        Update the weights and biases of the neural network using the averaged cost derivatives.
        :param avg_cost_derivatives_wrt_weights: List of averaged cost derivatives NumPy arrays with respect to weights.
        :param avg_cost_derivatives_wrt_biases: List of averaged cost derivatives NumPy arrays with respect to biases.
        :param learning_rate: Learning rate for gradient descent.
        """
        # Optimize weight gradients
        optimized_weight_step_arrays = self.weight_optimizer.step(avg_cost_derivatives_wrt_weights, learning_rate)
        # Update weights
        for layer in range(len(self.weights)):
            self.weights[layer] = (np.array(self.weights[layer]) - optimized_weight_step_arrays[layer]).tolist()
        # Optimize bias gradients
        optimized_bias_step_arrays = self.bias_optimizer.step(avg_cost_derivatives_wrt_biases, learning_rate)
        # Update biases
        for layer in range(len(self.biases)):
            self.biases[layer] = (np.array(self.biases[layer]) - optimized_bias_step_arrays[layer]).tolist()

    def train(self, training_data, activation_algos, epochs=100, learning_rate=0.01, decay_rate=0.9, dropout_rate=0.2):
        """
        Train the neural network using the provided training data.

        :param training_data: List of tuples [(activation_vector, target_vector), ...].
        :param activation_algos: Algorithms used to activate
        :param epochs: Number of training iterations.
        :param learning_rate: Learning rate for gradient descent.
        :param decay_rate: Decay rate of learning rate for finer gradient descent
        :param dropout_rate: Fraction of neurons to drop during training for hidden layers
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
            avg_cost_derivatives_wrt_weights = [np.zeros_like(np.array(w)) for w in self.weights]
            avg_cost_derivatives_wrt_biases = [np.zeros_like(np.array(b)) for b in self.biases]
            total_cost = 0

            for activation_vector, target_vector in training_data_sample:
                _, cost, cost_derivatives_wrt_weights, cost_derivatives_wrt_biases = self.compute_output(
                    activation_vector, activation_algos, target_vector, dropout_rate
                )
                total_cost += cost
                for layer in range(len(self.weights)):
                    avg_cost_derivatives_wrt_weights[layer] += np.array(cost_derivatives_wrt_weights[layer])
                    avg_cost_derivatives_wrt_biases[layer] += np.array(cost_derivatives_wrt_biases[layer])

            # Average the derivatives
            avg_cost_derivatives_wrt_weights = [w / len(training_data_sample) for w in avg_cost_derivatives_wrt_weights]
            avg_cost_derivatives_wrt_biases = [b / len(training_data_sample) for b in avg_cost_derivatives_wrt_biases]

            # Update weights and biases
            current_learning_rate = learning_rate * (decay_rate ** epoch)
            self._train_step(avg_cost_derivatives_wrt_weights, avg_cost_derivatives_wrt_biases, current_learning_rate)

            # Record progress
            self.progress.append({
                "dt": dt.now().isoformat(),
                "epoch": epoch + 1,
                "cost": total_cost / len(training_data_sample)
            })
            last_progress = self.progress[-1]
            print(f"Model {self.model_id}: {last_progress["dt"]} - Epoch {last_progress["epoch"]}, Cost: {last_progress["cost"]:.4f} ")

            # Serialize model after 10 secs while training
            if time.time() - last_serialized >= 10:
                self.serialize()

        # Serialize model after training
        self.serialize()
