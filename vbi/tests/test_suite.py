import os
import unittest
import importlib


def tests():
    test_directory = os.path.dirname(__file__)
    test_modules = [file[:-3] for file in os.listdir(test_directory) 
                    if file.startswith("test_") and file.endswith(".py")]
    test_modules = [module for module in test_modules if module != "test_suite.py"]

    suite = unittest.TestSuite()
    for module_name in test_modules:
        module = importlib.import_module(f".{module_name}", package=__package__)
        tests = unittest.TestLoader().loadTestsFromModule(module)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=0)
    runner.run(suite)
