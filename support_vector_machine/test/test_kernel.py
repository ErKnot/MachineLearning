from support_vector_machine.svm import linear_kernel

import unittest
import numpy as np

class TestKernel(unittest.TestCase):
    def test_kernel_vectors(self):
        v = np.array([1,2])
        w = np.array([3,4])
        correct_result = float(11)
        self.assertEqual(linear_kernel(v,w), correct_result, 'The kernel of two vectors is wrong')

    def test_kenel_matrices(self):
        v = np.array(np.arange(0,4)).reshape(2,2)
        w = np.array(np.arange(0,4)).reshape(2,2)
        correct_result = np.array([1,3,3,13]).reshape(2,2)
        self.assertTrue(np.array_equal(linear_kernel(v,w), correct_result), 'The kernel of two matrices is wrong.')

        



if __name__=="__main__":
    unittest.main()
    

