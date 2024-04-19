import unittest
import numpy as np 

class test_module_multiply(unittest.TestCase):
    def test_add(self):
        self.assertEqual(np.multiply(1, 2), 2)