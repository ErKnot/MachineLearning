import numpy as np
from multiclass_classification.utilities import softmax

import unittest

class TestSoftmax(unittest.TestCase):
    def test_sum_col(self):
        X = np.array([[0,0,np.log(0.2)], [0,np.log(4), np.log(3)]])
        result = np.around(softmax(X), 4)
        expected_result = np.array([[0.5, 0.2, round(0.2/3.2, 4)], [0.5, 0.8, round(3/3.2, 4)]])
        self.assertTrue(np.array_equal(result, expected_result), "The array differ.")

    def test_sum_row(self):
        X = np.array([[0,0,np.log(0.2)], [0,np.log(4), np.log(3)]]).T
        result = np.around(softmax(X, axis=1), 4)
        expected_result = np.array([[0.5, 0.2, round(0.2/3.2, 4)], [0.5, 0.8, round(3/3.2, 4)]]).T
        self.assertTrue(np.array_equal(result, expected_result), "The array differ.")


if __name__ == "__main__":
    unittest.main()
