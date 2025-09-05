import os
import unittest
import importlib
import argparse
import sys


def run_tests(category=None):
    """
    Run tests with optional category filtering.
    
    Args:
        category (str): Test category to run ('short', 'long', 'fast', 'slow', or None for all)
    """
    test_directory = os.path.dirname(__file__)
    test_modules = [file[:-3] for file in os.listdir(test_directory) 
                    if file.startswith("test_") and file.endswith(".py")]
    test_modules = [module for module in test_modules if module != "test_suite"]

    suite = unittest.TestSuite()
    for module_name in test_modules:
        module = importlib.import_module(f".{module_name}", package=__package__)
        tests = unittest.TestLoader().loadTestsFromModule(module)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


def run_with_pytest(category=None):
    """
    Run tests using pytest with category filtering.
    
    Args:
        category (str): Test category to run ('short', 'long', 'fast', 'slow', or None for all)
    """
    import subprocess
    import sys
    
    test_dir = os.path.dirname(__file__)
    cmd = [sys.executable, "-m", "pytest", test_dir, "-v"]
    
    if category:
        if category.lower() in ['short', 'fast']:
            cmd.extend(["-m", "short"])
        elif category.lower() in ['long', 'slow']:
            cmd.extend(["-m", "long"])
        elif category == 'not-long':
            cmd.extend(["-m", "not long"])
        elif category == 'not-short':
            cmd.extend(["-m", "not short"])
        else:
            print(f"Warning: Unknown category '{category}'. Running all tests.")
    
    # Add duration reporting
    cmd.extend(["--durations=10"])
    
    print(f"Running command: {' '.join(cmd)}")
    print("Note: For best results, run tests in the 'vbidevelop' conda environment")
    print("      to ensure all dependencies are available and tests don't get skipped.")
    print()
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VBI test suite")
    parser.add_argument(
        "--category", 
        choices=["short", "long", "fast", "slow", "not-long", "not-short", "all"],
        default="all",
        help="Test category to run (default: all)"
    )
    parser.add_argument(
        "--use-pytest", 
        action="store_true",
        help="Use pytest instead of unittest (recommended for category filtering)"
    )
    
    args = parser.parse_args()
    
    category = None if args.category == "all" else args.category
    
    if args.use_pytest or category:
        # Use pytest for category filtering
        if not args.use_pytest and category:
            print("Note: Using pytest for category filtering. Add --use-pytest to suppress this message.")
        exit_code = run_with_pytest(category)
        sys.exit(exit_code)
    else:
        # Use unittest runner
        result = run_tests(category)
        sys.exit(0 if result.wasSuccessful() else 1)
