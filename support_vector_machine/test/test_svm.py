import numpy as np
import unittest

from support_vector_machine.svm import SVM

class test_lin_kernel(unittest.TestCase):
    def test_dim_mult_vects(self):
        vec_1 = np.array([1,2,-3,0.5])
        vec_2 = np.array([0,-2, 0, 1])
        result = SVM.linear_kernel(vec_1, vec_2)
        self.assertIsInstance(result, float, "Wrong vector multiplication type.")

    def test_vectors(self):
        vec_1 = np.array([1,2,-3,0.5])
        vec_2 = np.array([0,-2, 0, 1])
        result = SVM.linear_kernel(vec_1, vec_2)
        correct_result = - 3.5
        self.assertEqual(result, correct_result, "Wrong vector multiplication.")

    def test_matrices(self):
        mat_1 = np.random.rand(2, 5)
        mat_2 = np.random.rand(2, 5)
        result = SVM.linear_kernel(mat_1, mat_2)
        self.assertEqual(result.shape, (2, 2), "Wong matrix multiplication dimension.")
# class test_svm_boundaries(unittest.TestCase):
#     def test_equal_targets(self):
#         svm = SVM()
#         result = svm.compute_boundaries(0.5, 2, 1, 1, 0.3)
#         correct_result = (2.2, 0.3)
#         self.assertEqual(result, correct_result, "Wrong boundaries")
#
#     def test_different_targets(self):
#         svm = SVM()
#         result = svm.compute_boundaries(0.5, 2, 1, -1, 0.3)
#         correct_result = (1.5, 0.3)
#         self.assertEqual(result, correct_result, "Wrong boundaries")
#
#     def test_equal_boundaries(self):
#         svm = SVM()
#         result = svm.compute_boundaries(0.5, 0.5, 1, -1, 0)
#         correct_result = 0
#         self.assertEqual(result, correct_result, "Doesn't detect equal boundaries")
#
#
# class test_compute_eta(unittest.TestCase):
#     def test_result(self):
#         x_1 = np.array([1,2], dtype="float64")
#         x_2 = np.array([3,4], dtype="float64")
#         svm = SVM()
#         result = svm.compute_eta(linear_kernel, x_1, x_2)
#         correct_result = 8
#         self.assertEqual(result, correct_result, "Eta is not correct.")
#
#
#

if __name__=="__main__":
    unittest.main()
