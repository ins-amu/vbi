#!/usr/bin/env python3
"""
Quick Coverage Check for VBI

A simple script to quickly identify modules without tests.
Integrates with the existing VBI test infrastructure.

Usage:
    python quick_coverage.py              # Basic coverage check
    python quick_coverage.py --missing    # Show only missing tests
    python quick_coverage.py --create     # Create template test files
"""

import os
import sys
from pathlib import Path
import argparse


def find_vbi_modules():
    """Find all Python modules in VBI package (excluding tests)."""
    vbi_root = Path(__file__).parent.parent  # Go up from tests/ to vbi/
    modules = []
    
    skip_dirs = {'tests', '__pycache__', '.pytest_cache', 'build', 'output'}
    skip_files = {'setup.py', 'conftest.py'}
    
    for py_file in vbi_root.rglob("*.py"):
        # Skip excluded directories
        if any(skip in py_file.parts for skip in skip_dirs):
            continue
            
        # Skip excluded files  
        if py_file.name in skip_files:
            continue
            
        # Skip some private files (but keep __init__.py)
        if py_file.name.startswith('_') and py_file.name != '__init__.py':
            continue
            
        modules.append(py_file)
    
    return sorted(modules)


def find_test_files():
    """Find all existing test files."""
    tests_dir = Path(__file__).parent
    test_files = list(tests_dir.glob("test_*.py"))
    return [f.stem for f in test_files]  # Return just the names without .py


def check_coverage():
    """Check which modules have corresponding test files."""
    modules = find_vbi_modules()
    test_files = find_test_files()
    
    print("VBI Test Coverage Check")
    print("=" * 50)
    
    total_modules = len(modules)
    tested_modules = 0
    missing_tests = []
    
    for module_path in modules:
        # Convert module path to expected test file name
        module_name = module_path.stem
        expected_test = f"test_{module_name}"
        
        # Check if there's a corresponding test file
        has_test = any(expected_test in test_file for test_file in test_files)
        
        if has_test:
            tested_modules += 1
            status = "✅"
        else:
            missing_tests.append((module_name, module_path))
            status = "❌"
            
        print(f"{status} {module_path.relative_to(module_path.parents[1])}")
    
    coverage_percent = (tested_modules / total_modules) * 100 if total_modules > 0 else 0
    
    print(f"\nSummary:")
    print(f"Total modules: {total_modules}")
    print(f"Modules with tests: {tested_modules}")
    print(f"Coverage: {coverage_percent:.1f}%")
    print(f"Missing tests: {len(missing_tests)}")
    
    return missing_tests


def show_missing_only():
    """Show only modules that are missing tests."""
    modules = find_vbi_modules()
    test_files = find_test_files()
    
    print("Modules Missing Tests:")
    print("-" * 30)
    
    missing_count = 0
    for module_path in modules:
        module_name = module_path.stem
        expected_test = f"test_{module_name}"
        
        has_test = any(expected_test in test_file for test_file in test_files)
        
        if not has_test:
            missing_count += 1
            rel_path = module_path.relative_to(module_path.parents[1])
            print(f"{missing_count:2d}. {rel_path}")
            print(f"    → Create: test_{module_name}.py")
    
    if missing_count == 0:
        print("🎉 All modules have test files!")
    else:
        print(f"\nTotal missing: {missing_count} test files")


def create_test_template(module_name, module_path):
    """Create a basic test template for a module."""
    template = f'''"""Tests for {module_name} module."""

import pytest
import numpy as np
from pathlib import Path

# Import the module being tested
import vbi.{module_name if module_name != '__init__' else ''}


class Test{module_name.title().replace('_', '')}:
    """Test class for {module_name} module."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        pass
    
    def teardown_method(self):
        """Clean up after each test method."""
        pass
    
    def test_imports(self):
        """Test that the module can be imported without errors."""
        # This test ensures the module loads correctly
        assert True  # Replace with actual import test
    
    def test_basic_functionality(self):
        """Test basic functionality of the module."""
        # TODO: Add specific tests for the main functions/classes
        pytest.skip("Test not implemented yet")
    
    # Add more test methods as needed
    # def test_specific_function(self):
    #     """Test a specific function."""
    #     pass
'''
    
    return template


def create_missing_tests():
    """Create template test files for modules missing tests."""
    missing_tests = []
    modules = find_vbi_modules()
    test_files = find_test_files()
    
    for module_path in modules:
        module_name = module_path.stem
        expected_test = f"test_{module_name}"
        
        has_test = any(expected_test in test_file for test_file in test_files)
        
        if not has_test:
            missing_tests.append((module_name, module_path))
    
    if not missing_tests:
        print("🎉 All modules already have test files!")
        return
    
    print(f"Creating {len(missing_tests)} test file templates...")
    tests_dir = Path(__file__).parent
    
    for module_name, module_path in missing_tests:
        test_file_path = tests_dir / f"test_{module_name}.py"
        
        if test_file_path.exists():
            print(f"⚠️  Skipping {test_file_path.name} (already exists)")
            continue
        
        template = create_test_template(module_name, module_path)
        
        with open(test_file_path, 'w') as f:
            f.write(template)
        
        print(f"✅ Created {test_file_path.name}")
    
    print(f"\nDone! Created test templates in {tests_dir}")
    print("Remember to:")
    print("1. Implement the actual test logic")
    print("2. Add appropriate test markers (@pytest.mark.short or @pytest.mark.long)")
    print("3. Run the tests: python run_tests.py")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quick VBI test coverage check")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--missing", action="store_true",
                      help="Show only modules missing tests")
    group.add_argument("--create", action="store_true",
                      help="Create template test files for missing tests")
    
    args = parser.parse_args()
    
    if args.missing:
        show_missing_only()
    elif args.create:
        create_missing_tests()
    else:
        check_coverage()


if __name__ == "__main__":
    main()
