import unittest
from parameterized import parameterized
import numpy as np
import functions as func

class TestValueError(unittest.TestCase):

    @parameterized.expand([
        (np.array([np.nan, 1, -1]),),
        (np.array([np.inf, 1, -1]),),
    ])
    def test_tanh_value_error(self, x):
        with self.assertRaises(ValueError):
            func.tanh(x)

    @parameterized.expand([
        (np.array([np.nan, 1, -1]),),
        (np.array([np.inf, 1, -1]),),
    ])
    def test_softmax_value_error(self, x):
        with self.assertRaises(ValueError):
            func.softmax(x)

if __name__ == "__main__":
    unittest.main()
