#!/usr/bin/env python3
"""
VBI Test Runner

This script provides an easy way to run VBI tests with different categories.

Usage examples:
    python run_tests.py --short          # Run only fast/short tests
    python run_tests.py --long           # Run only slow/long tests  
    python run_tests.py --all            # Run all tests (default)
    python run_tests.py --not-long       # Run all tests except slow ones
    
You can also run tests directly with pytest:
    pytest . -m short                    # Fast tests only
    pytest . -m long                     # Slow tests only
    pytest . -m "not long"               # All except slow tests
    pytest . --durations=10              # Show 10 slowest tests

This is a permanent file that should be kept in the repository.
"""

import argparse
import sys
import subprocess
import os


def main():
    parser = argparse.ArgumentParser(
        description="Run VBI tests with category filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--short", "--fast", action="store_true", 
                       help="Run only fast/short tests")
    group.add_argument("--long", "--slow", action="store_true",
                       help="Run only slow/long tests")
    group.add_argument("--not-long", action="store_true",
                       help="Run all tests except slow ones (good for CI)")
    group.add_argument("--not-short", action="store_true", 
                       help="Run all tests except fast ones")
    group.add_argument("--all", action="store_true", default=True,
                       help="Run all tests (default)")
    
    parser.add_argument("--durations", type=int, default=10,
                       help="Show N slowest tests (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet output")
    
    args = parser.parse_args()
    
    # Determine the marker
    if args.short:
        marker = "short"
    elif args.long:
        marker = "long"
    elif args.not_long:
        marker = "not long"
    elif args.not_short:
        marker = "not short"
    else:
        marker = None
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", "."]  # Current directory (tests/)
    
    if marker:
        cmd.extend(["-m", marker])
    
    if args.verbose:
        cmd.append("-v")
    elif args.quiet:
        cmd.append("-q")
    else:
        cmd.append("-v")  # Default to verbose
    
    if args.durations > 0:
        cmd.extend(["--durations", str(args.durations)])
    
    # Add other useful options
    cmd.extend(["--tb=short"])  # Short traceback format
    
    # Change to the tests directory
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(tests_dir)
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Run the tests
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
