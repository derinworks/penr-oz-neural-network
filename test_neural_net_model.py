import unittest
from parameterized import parameterized
from neural_net_model import NeuralNetworkModel

class TestNeuralNetModel(unittest.TestCase):

    @parameterized.expand([
        ([9, 9, 9], "xavier", "random",),
        ([18, 9, 3], "xavier", "zeros",),
        ([9, 18, 9], "he", "zeros",),
        ([4, 8, 16], "he", "random",),
        ([3, 3, 3, 3], "gaussian", "random",),
    ])
    def test_model_initialization(self, layer_sizes, weight_algo, bias_algo):
        model = NeuralNetworkModel(model_id="test", layer_sizes=layer_sizes, weight_algo=weight_algo,
                                   bias_algo=bias_algo)

        # Test if the model initializes correctly
        self.assertIsNotNone(model)
        # Check the number of weight matrices
        expected_weights = len(layer_sizes) - 1
        self.assertEqual(len(model.weights), expected_weights)

        # Check the dimensions of each weight matrix and bias vector
        for i, (weights, biases) in enumerate(zip(model.weights, model.biases)):
            self.assertEqual(len(weights), layer_sizes[i])  # Number of rows in weight matrix
            self.assertEqual(len(weights[0]), layer_sizes[i + 1])  # Number of columns in weight matrix
            self.assertEqual(len(biases), layer_sizes[i + 1])  # Bias vector length matches next layer size

        # Check the training buffer size
        expected_buffer_size = sum(
            layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )
        expected_sample_size = int(expected_buffer_size * 0.1)
        self.assertEqual(model.training_buffer_size, expected_buffer_size)
        self.assertEqual(model.training_sample_size, expected_sample_size)

    @parameterized.expand([
        ([9, 9, 9], ["sigmoid"] * 2,),
        ([18, 9, 3], ["relu", "softmax"],),
        ([9, 18, 9], ["tanh"] * 2,),
        ([4, 8, 16], ["softmax"] * 2,),
        ([3, 3, 3, 3], ["relu", "relu", "softmax"],),
    ])
    def test_compute_output(self, layer_sizes, algos):
        model = NeuralNetworkModel(model_id="test", layer_sizes=layer_sizes)

        # Verify that the model produces outputs of the expected shape
        input_size = layer_sizes[0]  # Size of the input vector
        output_size = layer_sizes[-1]  # Size of the output vector based on the final layer

        sample_input = [0.5] * input_size  # Example input as a list of numbers

        # Call the compute_output method
        output, _, _ = model.compute_output(activation_vector=sample_input, algos=algos)

        # Check that the output has the correct size
        self.assertEqual(len(output), output_size)

        # Optionally, you can verify the range or type of the output values
        for value in output:
            self.assertIsInstance(value, float)  # Ensure output values are floats

    @parameterized.expand([
        ([9, 9, 9], ["sigmoid"] * 2,),
        ([9, 9, 9], ["relu", "softmax"],),
        ([9, 9, 9], ["tanh"] * 2,),
        ([18, 9, 3], ["relu", "sigmoid"],),
        ([9, 18, 9], ["sigmoid", "softmax"],),
        ([4, 8, 16], ["sigmoid"] * 2,),
        ([3, 3, 3, 3], ["relu", "relu", "softmax"],),
        ([18, 9, 3], ["relu"] * 2,),
        ([9, 18, 9], ["relu", "tanh"],),
    ])
    def test_train(self, layer_sizes, algos):
        model = NeuralNetworkModel(model_id="test", layer_sizes=layer_sizes)

        # Check if training step updates the model
        input_size = layer_sizes[0]
        output_size = layer_sizes[-1]

        sample_input = [0.5] * input_size  # Example input as a list of numbers
        sample_target = [0.0] * output_size  # Example target as a list of numbers
        sample_target[0] = 1.0

        initial_weights = [layer_weights.tolist() for layer_weights in model.weights]
        initial_biases = [layer_biases.tolist() for layer_biases in model.biases]

        # Add enough data to meet the training buffer size
        training_data = [(sample_input, sample_target)] * model.training_buffer_size

        model.train(training_data=training_data, algos=algos, epochs=1)

        updated_weights = [layer_weights.tolist() for layer_weights in model.weights]
        updated_biases = [layer_biases.tolist() for layer_biases in model.biases]

        # Check that the model data is still valid
        self.assertEqual(len(initial_weights), len(updated_weights))
        self.assertEqual(len(initial_biases), len(updated_biases))

        # Ensure training progress
        self.assertGreater(len(model.progress), 0)
        self.assertEqual(len(model.training_data_buffer), 0)

        # Deserialize and check if recorded training
        persisted_model = NeuralNetworkModel.deserialize(model.model_id)

        persisted_weights = [layer_weights.tolist() for layer_weights in persisted_model.weights]
        persisted_biases = [layer_biases.tolist() for layer_biases in persisted_model.biases]

        self.assertEqual(updated_weights, persisted_weights)
        self.assertEqual(updated_biases, persisted_biases)
        self.assertEqual(len(persisted_model.progress), len(model.progress))
        self.assertEqual(len(persisted_model.training_data_buffer), 0)

    def test_train_with_insufficient_data(self):
        model = NeuralNetworkModel(model_id="test", layer_sizes=[9, 9, 9])

        # Test that training does not proceed when data is less than the buffer size
        input_size = 9
        output_size = 9

        sample_input = [0.5] * input_size  # Example input as a list of numbers
        sample_target = [1.0] * output_size  # Example target as a list of numbers

        # Add insufficient data
        training_data = [(sample_input, sample_target)] * (model.training_buffer_size - 1)

        model.train(training_data=training_data, algos=["relu"] * 2, epochs=1)

        # Ensure no training progress and buffering
        self.assertEqual(len(model.progress), 0)
        self.assertGreaterEqual(len(model.training_data_buffer), len(training_data))

        # Deserialize and check if recorded training buffer
        persisted_model = NeuralNetworkModel.deserialize(model.model_id)

        self.assertEqual(len(persisted_model.training_data_buffer), len(model.training_data_buffer))

    def test_invalid_activation_algo(self):
        model = NeuralNetworkModel(model_id="test", layer_sizes=[9, 9, 9])

        input_size = 9
        output_size = 9

        sample_input = [0.5] * input_size  # Example input as a list of numbers
        sample_target = [1.0] * output_size  # Example target as a list of numbers

        # Add enough data to meet the training buffer size
        training_data = [(sample_input, sample_target)] * model.training_buffer_size

        # Test that setting an invalid activation algorithm raises a ValueError
        with self.assertRaises(ValueError) as context:
            model.train(training_data=training_data, algos=["unknown_algo"] * 2, epochs=1)

        # Assert the error message
        self.assertEqual(str(context.exception), "Unsupported activation algorithm: unknown_algo")

    def test_invalid_model_deserialization(self):
        # Test that deserializing a nonexistent model raises a KeyError
        with self.assertRaises(KeyError):
            NeuralNetworkModel.deserialize("nonexistent_model")

    def test_delete(self):
        NeuralNetworkModel.delete("test")
        with self.assertRaises(KeyError):
            NeuralNetworkModel.deserialize("test")

    def test_invalid_delete(self):
        # No error raised for failing to delete
        NeuralNetworkModel.delete("nonexistent")

if __name__ == '__main__':
    unittest.main()
