import unittest
from neural_net_model import NeuralNetworkModel

class TestNeuralNetModel(unittest.TestCase):
    def setUp(self):
        # Initialize the model and any required components
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
        self.assertEqual(len(self.model.biases), 2)  # Two weight matrices connecting three layers
        # Check the dimensions of each bias vector
        for layer_bias_vector in self.model.biases:
            self.assertEqual(len(layer_bias_vector), 9)

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
            self.assertGreaterEqual(value, 0.0)  # Assuming non-negative outputs (e.g., ReLU activation)

    def test_train(self):
        # Check if a single training step updates the model's weights
        input_size = 9  # Replace with actual input size of your model
        output_size = 9  # Replace with actual output size of your model

        sample_input = [0.5] * input_size  # Example input as a list of numbers
        sample_target = [1.0] * output_size  # Example target as a list of numbers

        initial_weights = [layer_weights for layer_weights in self.model.weights]

        # Assuming training updates weights
        self.model.train(training_data=[(sample_input, sample_target)], epochs=1)

        updated_weights = [layer_weights for layer_weights in self.model.weights]

        # Check that weights have been updated
        self.assertNotEqual(initial_weights, updated_weights)

if __name__ == '__main__':
    unittest.main()
