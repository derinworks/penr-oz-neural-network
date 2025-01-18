import unittest
from parameterized import parameterized
from neural_net_model import NeuralNetworkModel

class TestNeuralNetModel(unittest.TestCase):
    def setUp(self):
        self.model = NeuralNetworkModel(model_id="test", layer_sizes=[9, 9, 9])

    def test_model_initialization(self):
        # Test if the model initializes correctly
        self.assertIsNotNone(self.model)
        # Check the number of weight matrices
        self.assertEqual(len(self.model.weights), 2)  # Two weight matrices connecting three layers
        # Check the dimensions of each weight matrix
        for layer_weights in self.model.weights:
            self.assertEqual(len(layer_weights), 9)
            for layer_weight_vector in layer_weights:
                self.assertEqual(len(layer_weight_vector), 9)
        # Check the number of bias vectors
        self.assertEqual(len(self.model.biases), 2)  # Two bias vectors connecting three layers
        # Check the dimensions of each bias vector
        for layer_bias_vector in self.model.biases:
            self.assertEqual(len(layer_bias_vector), 9)

        # Check the training buffer size
        self.assertEqual(self.model.training_buffer_size, int(0.1 * ((9 * 9 + 9) + (9 * 9 + 9))))

    def test_compute_output(self):
        # Verify that the model produces outputs of the expected shape
        input_size = 9  # Size of the input vector
        output_size = 9  # Size of the output vector based on the final layer

        sample_input = [0.5] * input_size  # Example input as a list of numbers

        # Call the compute_output method
        output, _, _, _ = self.model.compute_output(sample_input)

        # Check that the output has the correct size
        self.assertEqual(len(output), output_size)

        # Optionally, you can verify the range or type of the output values
        for value in output:
            self.assertIsInstance(value, float)  # Ensure output values are floats

    @parameterized.expand([
        ("relu",),
        ("sigmoid",),
        ("tanh",)
    ])
    def test_train(self, algo):
        # Check if training step updates the model
        input_size = 9
        output_size = 9

        sample_input = [0.5] * input_size  # Example input as a list of numbers
        sample_target = [1.0] * output_size  # Example target as a list of numbers

        initial_weights = [layer_weights for layer_weights in self.model.weights]
        initial_biases = [layer_biases for layer_biases in self.model.biases]

        # Add enough data to meet the training buffer size
        training_data = [(sample_input, sample_target)] * self.model.training_buffer_size

        self.model.train(training_data=training_data, activation_algo=algo, epochs=1)

        updated_weights = [layer_weights for layer_weights in self.model.weights]
        updated_biases = [layer_biases for layer_biases in self.model.biases]

        # Check that the model data is still valid
        self.assertEqual(len(initial_weights), len(updated_weights))
        self.assertEqual(len(initial_biases), len(updated_biases))

        # Deserialize and check if recorded training
        persisted_model = NeuralNetworkModel.deserialize(self.model.model_id)

        persisted_weights = [layer_weights for layer_weights in persisted_model.weights]
        persisted_biases = [layer_biases for layer_biases in persisted_model.biases]

        self.assertEqual(updated_weights, persisted_weights)
        self.assertEqual(updated_biases, persisted_biases)
        self.assertGreater(len(persisted_model.progress), 0)

    def test_train_with_insufficient_data(self):
        # Test that training does not proceed when data is less than the buffer size
        input_size = 9
        output_size = 9

        sample_input = [0.5] * input_size  # Example input as a list of numbers
        sample_target = [1.0] * output_size  # Example target as a list of numbers

        # Add insufficient data
        training_data = [(sample_input, sample_target)] * (self.model.training_buffer_size - 1)

        initial_weights = [layer_weights for layer_weights in self.model.weights]
        initial_biases = [layer_biases for layer_biases in self.model.biases]

        self.model.train(training_data=training_data, activation_algo="relu", epochs=1)

        updated_weights = [layer_weights for layer_weights in self.model.weights]
        updated_biases = [layer_biases for layer_biases in self.model.biases]

        # Ensure weights and biases have not been updated
        self.assertEqual(initial_weights, updated_weights)
        self.assertEqual(initial_biases, updated_biases)

    def test_invalid_activation_algo(self):
        # Test that setting an invalid activation algorithm raises a ValueError
        with self.assertRaises(ValueError) as context:
            input_size = 9
            output_size = 9

            sample_input = [0.5] * input_size  # Example input as a list of numbers
            sample_target = [1.0] * output_size  # Example target as a list of numbers

            # Add enough data to meet the training buffer size
            training_data = [(sample_input, sample_target)] * self.model.training_buffer_size

            self.model.train(training_data=training_data,
                             activation_algo="unknown_algo", epochs=1)

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
