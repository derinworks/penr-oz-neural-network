import numpy as np
import unittest
from parameterized import parameterized
import functions as func

class TestFunctions(unittest.TestCase):

    @parameterized.expand([
        (func.sigmoid(np.array([-501, -499, -1.5, 0.0, 1.5, 499, 501])), [0.0, 0.0, 0.1824, 0.5, 0.8176, 1.0, 1.0]),
        (func.sigmoid_derivative(np.array([0.0, 0.0, 0.1824, 0.5, 0.8176, 1.0, 1.0])), [0.0, 0.0, 0.1491, 0.25, 0.1491, 0.0, 0.0]),
        (func.relu(np.array([-1.5, 0.0, 1.5])), [0.0, 0.0, 1.5]),
        (func.relu_derivative(np.array([0.0, 0.0, 1.5])), [0.0, 0.0, 1.0]),
        (func.tanh(np.array([-1.5, 0.0, 1.5])), [-0.9051, 0.0, 0.9051]),
        (func.tanh_derivative(np.array([-0.9051, 0.0, 0.9051])), [0.1808, 1.0, 0.1808]),
        (func.mean_squared_error(np.array([-1, 1]), np.array([0, 1])), 0.5),
        (func.mean_squared_error(np.array([0, 0]), np.array([1, 0])), 0.5),
        (func.mean_squared_error(np.array([1.5, 1.0]), np.array([0.0, 0.7])), 1.17),
        (func.cross_entropy_loss(np.array([0.2, 0.3, 0.5]), np.array([0.3, 0.4, 0.4])), 0.4139),
        (func.cross_entropy_loss(np.array([-1.5, 0.2, 2.0]), np.array([0.3, 0.4, 0.4])), 2.9777),
        (func.cross_entropy_gradient(np.array([0.2, 0.3, 0.5]), np.array([0.3, 0.4, 0.3])), [0.1, 0.1, -0.2]),
        (func.softmax(np.array([-1.5, 0.2, 2.0])), [0.0253, 0.1383, 0.8365]),
        (func.softmax(np.array([0.0, 0.0, 0.0])), [0.3333, 0.3333, 0.3333]),
        (func.softmax(np.array([1.5, 0.3, -2.0])), [0.7511, 0.2262, 0.0227]),
        (func.batch_norm(np.array([-1.5, 0.2, 2.0])), [-1.2129, -0.0233, 1.2362]),
        (func.batch_norm(np.array([0.0, 0.0, 0.0])), [0.0, 0.0, 0.0]),
        (func.apply_dropout(np.array([-1.5, 0.2, 2.0]), 0.0), [-1.5, 0.2, 2.0]),
        (func.apply_dropout(np.array([-1.5, 0.2, 2.0]), 0.99999), [0.0, 0.0, 0.0]),
        (func.apply_dropout(np.array([-1.5, 0.2, 2.0]), 1.0), [0.0, 0.0, 0.0]),
    ])
    def test_func(self, actual: np.ndarray[float], expected: np.ndarray[float]):
        np.testing.assert_array_almost_equal(expected, actual, 4)

    @parameterized.expand([
        (np.nan, func.tanh, ValueError),
        (np.inf, func.tanh, ValueError),
        ([np.nan, 1, -1], func.softmax, ValueError),
        ([np.inf, 1, -1], func.softmax, ValueError),
    ])
    def test_func_error(self, x, f, expected):
        with self.assertRaises(expected):
            f(np.array(x))

if __name__ == "__main__":
    unittest.main()
