import unittest
import numpy as np
from parameterized import parameterized
from adam_optimizer import AdamOptimizer

class TestAdamOptimizer(unittest.TestCase):

    def setUp(self):
        # Initialize optimizer
        self.optimizer = AdamOptimizer()

    @parameterized.expand([
        ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
         [[0.001, 0.001, 0.001], [0.001, 0.001, 0.001]]),
        ([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
         [[[0.001, 0.001, 0.001], [0.001, 0.001, 0.001]], [[0.001, 0.001, 0.001], [0.001, 0.001, 0.001]]]),
    ])
    def test_adam_optimizer(self, gradients, expected_steps):
        gradient_arrays = [gradient_array for gradient_array in map(np.array, gradients)]

        step_arrays = self.optimizer.step(gradient_arrays)

        self.assertEqual(self.optimizer.state["time_step"], 1)
        for step_array, expected_step_array in zip(step_arrays, map(np.array, expected_steps)):
            np.testing.assert_array_almost_equal(step_array, expected_step_array, decimal=8)

if __name__ == '__main__':
    unittest.main()
