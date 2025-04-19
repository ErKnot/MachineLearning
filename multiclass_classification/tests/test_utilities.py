import unittest

from multiclass_classification.utilities import onehotencoder
import numpy as np

class TestOnehotencoder(unittest.TestCase):
    def test_list_numbers(self):
        """
        Test if it can handle a list of numbers
        """
        data = [2, 1, 2]
        expected_result = np.array([[0,1], [1,0], [0,1]])
        result = onehotencoder(data)
        self.assertTrue((result==expected_result).all(), "The arrays differ.")

    def test_list_numbers_2(self):
        """
        Test if it can handle a list of numbers, giving a dictionary of classes
        """
        data = [2, 1, 2]
        expected_result = np.array([[0,1, 0], [0,0,1], [0,1,0]])
        classes_map = {0:0, 2:1, 1:2}
        result = onehotencoder(data, classes_map)
        self.assertTrue((result==expected_result).all(), "The arrays differ.")

    def test_list_strings(self):
        """
        Test a list of strings
        """
        data = ["orange", "banana"]
        expected_result = np.array([[0,1], [1,0]])
        result = onehotencoder(data)
        self.assertTrue((result==expected_result).all(), "The arrys differ." )


    def test_list_strings(self):
        """
        Test a list of strings
        """
        data = ["orange", "banana"]
        expected_result = np.array([[1,0,0], [0,0,1]])
        classes_map = {"orange": 0, "car": 1, "banana": 2}
        result = onehotencoder(data, classes_map)
        self.assertTrue((result==expected_result).all(), "The arrys differ." )
    

    def test_nparray(self):
        """
        Test a nparray
        """
        data = np.array([[1], [2], [0]])
        expected_result = np.array(([[0,1,0], [0,0,1], [1,0,0]]))
        result = onehotencoder(data)
        self.assertTrue((result==expected_result).all(), "The arrys differ." )




if __name__ == "__main__":
    unittest.main()
