#!/usr/bin/env python3
"""
VBI Test Coverage Checker

This script analyzes the VBI package and identifies modules, classes, and functions
that are missing test coverage. It helps identify what needs to be tested.

Usage:
    python coverage_checker.py                    # Show missing coverage
    python coverage_checker.py --detailed         # Show detailed analysis
    python coverage_checker.py --run-coverage     # Run pytest with coverage
    python coverage_checker.py --html-report      # Generate HTML coverage report

Requirements:
    pip install coverage pytest-cov
"""

import ast
import os
import sys
import importlib
import inspect
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import argparse


@dataclass
class ModuleInfo:
    """Information about a module."""
    name: str
    file_path: str
    classes: List[str]
    functions: List[str]
    has_tests: bool
    test_files: List[str]


@dataclass
class CoverageReport:
    """Coverage analysis report."""
    total_modules: int
    tested_modules: int
    total_functions: int
    tested_functions: int
    missing_tests: Dict[str, ModuleInfo]


class VBICoverageChecker:
    """Analyzes VBI package for test coverage."""
    
    def __init__(self, vbi_root: str = None):
        """Initialize the coverage checker.
        
        Args:
            vbi_root: Path to VBI root directory. If None, auto-detect.
        """
        if vbi_root is None:
            # Auto-detect VBI root
            current_dir = Path(__file__).parent
            self.vbi_root = current_dir.parent
        else:
            self.vbi_root = Path(vbi_root)
            
        self.tests_dir = self.vbi_root / "tests"
        self.package_dir = self.vbi_root
        
    def find_python_modules(self) -> List[Path]:
        """Find all Python modules in the VBI package."""
        modules = []
        
        # Skip certain directories and files
        skip_dirs = {'tests', '__pycache__', '.pytest_cache', 'build', 'dist', '.git'}
        skip_files = {'setup.py', 'conftest.py'}
        
        for py_file in self.package_dir.rglob("*.py"):
            # Skip if in excluded directories
            if any(skip_dir in py_file.parts for skip_dir in skip_dirs):
                continue
                
            # Skip excluded files
            if py_file.name in skip_files:
                continue
                
            # Skip private modules (starting with _) unless they're important
            if py_file.name.startswith('_') and py_file.name not in ['__init__.py', '_version.py']:
                continue
                
            modules.append(py_file)
            
        return sorted(modules)
    
    def analyze_module(self, module_path: Path) -> ModuleInfo:
        """Analyze a Python module to extract classes and functions."""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    # Skip private functions and methods inside classes
                    if not node.name.startswith('_'):
                        functions.append(node.name)
                        
        except Exception as e:
            print(f"Warning: Could not parse {module_path}: {e}")
            classes = []
            functions = []
            
        # Convert path to module name
        rel_path = module_path.relative_to(self.vbi_root)
        module_name = str(rel_path.with_suffix('')).replace(os.sep, '.')
        
        # Check for corresponding test files
        test_files = self.find_test_files(module_name, module_path)
        has_tests = len(test_files) > 0
        
        return ModuleInfo(
            name=module_name,
            file_path=str(module_path),
            classes=classes,
            functions=functions,
            has_tests=has_tests,
            test_files=test_files
        )
    
    def find_test_files(self, module_name: str, module_path: Path) -> List[str]:
        """Find test files that might test the given module."""
        test_files = []
        
        if not self.tests_dir.exists():
            return test_files
            
        # Look for test files with similar names
        module_basename = module_path.stem
        
        for test_file in self.tests_dir.rglob("test_*.py"):
            # Check if test file name suggests it tests this module
            if module_basename in test_file.stem or module_name.replace('.', '_') in test_file.stem:
                test_files.append(str(test_file))
                
        return test_files
    
    def generate_coverage_report(self) -> CoverageReport:
        """Generate a comprehensive coverage report."""
        modules = self.find_python_modules()
        
        total_modules = 0
        tested_modules = 0
        total_functions = 0
        tested_functions = 0
        missing_tests = {}
        
        print("Analyzing VBI package for test coverage...")
        print(f"Found {len(modules)} Python modules to analyze\n")
        
        for module_path in modules:
            module_info = self.analyze_module(module_path)
            
            total_modules += 1
            total_functions += len(module_info.functions)
            
            if module_info.has_tests:
                tested_modules += 1
                # For simplicity, assume all functions in tested modules are tested
                # A more sophisticated analysis would parse the test files
                tested_functions += len(module_info.functions)
            else:
                missing_tests[module_info.name] = module_info
                
        return CoverageReport(
            total_modules=total_modules,
            tested_modules=tested_modules,
            total_functions=total_functions,
            tested_functions=tested_functions,
            missing_tests=missing_tests
        )
    
    def print_coverage_summary(self, report: CoverageReport):
        """Print a summary of the coverage analysis."""
        print("=" * 60)
        print("VBI TEST COVERAGE SUMMARY")
        print("=" * 60)
        
        module_coverage = (report.tested_modules / report.total_modules) * 100 if report.total_modules > 0 else 0
        function_coverage = (report.tested_functions / report.total_functions) * 100 if report.total_functions > 0 else 0
        
        print(f"Modules with tests: {report.tested_modules}/{report.total_modules} ({module_coverage:.1f}%)")
        print(f"Functions with tests: {report.tested_functions}/{report.total_functions} ({function_coverage:.1f}%)")
        print()
        
        if report.missing_tests:
            print(f"MODULES MISSING TESTS ({len(report.missing_tests)}):")
            print("-" * 40)
            for module_name, module_info in report.missing_tests.items():
                print(f"📁 {module_name}")
                if module_info.classes:
                    print(f"   Classes: {', '.join(module_info.classes)}")
                if module_info.functions:
                    print(f"   Functions: {', '.join(module_info.functions[:5])}" + 
                          (f" ... ({len(module_info.functions)-5} more)" if len(module_info.functions) > 5 else ""))
                print()
        else:
            print("🎉 All modules have test files!")
            
    def print_detailed_report(self, report: CoverageReport):
        """Print a detailed coverage report."""
        self.print_coverage_summary(report)
        
        if report.missing_tests:
            print("=" * 60)
            print("DETAILED MISSING COVERAGE")
            print("=" * 60)
            
            for module_name, module_info in report.missing_tests.items():
                print(f"\n📦 Module: {module_name}")
                print(f"   File: {module_info.file_path}")
                
                if module_info.classes:
                    print(f"   Classes needing tests:")
                    for cls in module_info.classes:
                        print(f"     - {cls}")
                        
                if module_info.functions:
                    print(f"   Functions needing tests:")
                    for func in module_info.functions:
                        print(f"     - {func}")
                        
                print(f"   Suggested test file: test_{Path(module_info.file_path).stem}.py")
                print("-" * 40)
    
    def run_pytest_coverage(self, html_report: bool = False):
        """Run pytest with coverage reporting."""
        print("Running pytest with coverage analysis...")
        
        cmd = [
            sys.executable, "-m", "pytest", 
            str(self.tests_dir),
            "--cov=" + str(self.package_dir),
            "--cov-report=term-missing",
        ]
        
        if html_report:
            html_dir = self.vbi_root / "htmlcov"
            cmd.extend(["--cov-report=html:" + str(html_dir)])
            print(f"HTML report will be generated in: {html_dir}")
        
        print(f"Command: {' '.join(cmd)}")
        print()
        
        try:
            result = subprocess.run(cmd, cwd=str(self.vbi_root))
            if html_report and result.returncode == 0:
                print(f"\nHTML coverage report generated: {html_dir}/index.html")
                print(f"Open in browser: file://{html_dir}/index.html")
        except FileNotFoundError:
            print("Error: pytest or coverage not found. Install with:")
            print("pip install pytest pytest-cov")
    
    def suggest_test_structure(self, report: CoverageReport):
        """Suggest test file structure for missing coverage."""
        if not report.missing_tests:
            return
            
        print("=" * 60)
        print("SUGGESTED TEST FILES TO CREATE")
        print("=" * 60)
        
        for module_name, module_info in report.missing_tests.items():
            test_filename = f"test_{Path(module_info.file_path).stem}.py"
            print(f"\n📝 Create: {self.tests_dir}/{test_filename}")
            print(f"   To test: {module_name}")
            
            # Generate basic test template
            template = self.generate_test_template(module_info)
            print("   Template:")
            for line in template.split('\n')[:10]:  # Show first 10 lines
                print(f"     {line}")
            if len(template.split('\n')) > 10:
                print("     ...")
    
    def generate_test_template(self, module_info: ModuleInfo) -> str:
        """Generate a basic test template for a module."""
        module_import = module_info.name.replace(os.sep, '.')
        
        template = f'''"""Tests for {module_info.name}."""

import pytest
from {module_import} import *


class Test{Path(module_info.file_path).stem.title()}:
    """Test class for {module_info.name}."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        pass
    
    def teardown_method(self):
        """Tear down test fixtures after each test method."""
        pass
'''

        # Add test methods for each function
        for func in module_info.functions[:3]:  # Limit to first 3 functions
            template += f'''
    def test_{func}(self):
        """Test {func} function."""
        # TODO: Implement test for {func}
        pass
'''

        # Add test methods for each class
        for cls in module_info.classes[:2]:  # Limit to first 2 classes
            template += f'''
    def test_{cls.lower()}_creation(self):
        """Test {cls} class creation."""
        # TODO: Implement test for {cls}
        pass
'''

        return template


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze VBI test coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed coverage analysis")
    parser.add_argument("--run-coverage", action="store_true",
                       help="Run pytest with coverage reporting")
    parser.add_argument("--html-report", action="store_true",
                       help="Generate HTML coverage report")
    parser.add_argument("--suggest-tests", action="store_true",
                       help="Suggest test file structure")
    parser.add_argument("--vbi-root", type=str,
                       help="Path to VBI root directory (auto-detected if not provided)")
    
    args = parser.parse_args()
    
    # Create coverage checker
    checker = VBICoverageChecker(args.vbi_root)
    
    if args.run_coverage or args.html_report:
        checker.run_pytest_coverage(html_report=args.html_report)
        return
    
    # Generate coverage report
    report = checker.generate_coverage_report()
    
    if args.detailed:
        checker.print_detailed_report(report)
    else:
        checker.print_coverage_summary(report)
    
    if args.suggest_tests:
        checker.suggest_test_structure(report)


if __name__ == "__main__":
    main()
