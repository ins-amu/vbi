import os
import vbi
import glob
import unittest
from os.path import join
from unittest import TestLoader, TextTestRunner, TestSuite



def get_module_path():
    '''
    Returns the location of the tests folder
    '''
    tests_folder = "tests"
    location = vbi.__file__
    location = location.replace('__init__.py', '')
    location = join(location, tests_folder)

    return location


def tests():
    """
    Find all test_*.py files in the tests folder and run them

    """

    path = get_module_path()
    test_suite = unittest.TestLoader().discover(path, pattern='test_*.py')
    test_runner = TextTestRunner().run(test_suite)



if __name__ == '__main__':
    tests()
