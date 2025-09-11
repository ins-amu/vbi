# VBI Test Coverage Tools

This directory contains tools to help identify and improve test coverage for the VBI package.

## Current Test Coverage: 12.3% (8/65 modules)

## Tools Available

### 1. Quick Coverage Checker (`quick_coverage.py`)

A simple tool to quickly identify which modules are missing tests.

```bash
# Basic coverage overview
python quick_coverage.py

# Show only modules missing tests
python quick_coverage.py --missing

# Create template test files for missing modules
python quick_coverage.py --create
```

### 2. Enhanced Test Runner (`run_tests.py`)

The existing test runner now includes coverage checking capabilities.

```bash
# Check which modules are missing tests
python run_tests.py --check-missing

# Run tests with coverage reporting
python run_tests.py --coverage

# Generate HTML coverage report
python run_tests.py --html-coverage
```

### 3. Comprehensive Coverage Checker (`coverage_checker.py`)

A detailed analysis tool that provides in-depth coverage information.

```bash
# Basic coverage analysis
python coverage_checker.py

# Detailed analysis with function-level information
python coverage_checker.py --detailed

# Run pytest with coverage
python coverage_checker.py --run-coverage

# Generate HTML coverage report
python coverage_checker.py --html-report

# Show suggested test file structure
python coverage_checker.py --suggest-tests
```

## Installation

Install the required testing dependencies:

```bash
pip install -r test_requirements.txt
```

Or install individually:

```bash
pip install pytest pytest-cov coverage
```

## Current Status

### Modules WITH Tests (8):
- `test_features.py` → tests feature extraction
- `test_ghb_cupy.py` → tests GHB CUPY implementation  
- `test_mpr_cupy.py` → tests MPR CUPY implementation
- `test_mpr_numba.py` → tests MPR Numba implementation
- And 4 others...

### Modules MISSING Tests (57):
Major modules that need test coverage:
- `vbi/inference.py` - Core inference functionality
- `vbi/utils.py` - Utility functions
- `vbi/sbi_inference.py` - SBI inference methods
- `feature_extraction/calc_features.py` - Feature calculation
- All model implementations in various backends (CUPY, JAX, Numba, etc.)

## Workflow for Adding Tests

1. **Identify missing coverage:**
   ```bash
   python quick_coverage.py --missing
   ```

2. **Create test template:**
   ```bash
   python quick_coverage.py --create
   ```

3. **Edit the generated test file** and implement actual tests

4. **Run tests to verify:**
   ```bash
   python run_tests.py --coverage
   ```

5. **Generate detailed coverage report:**
   ```bash
   python run_tests.py --html-coverage
   ```

## Test File Naming Convention

- For module `vbi/inference.py` → create `test_inference.py`
- For module `models/cupy/mpr.py` → create `test_mpr.py` (if testing cupy-specific functionality)
- For module `feature_extraction/calc_features.py` → create `test_calc_features.py`

## Test Categories

Use pytest markers to categorize tests:

```python
@pytest.mark.short  # Fast tests (< 1 second)
@pytest.mark.long   # Slow tests (> 1 second)
```

Run specific categories:
```bash
python run_tests.py --short    # Fast tests only
python run_tests.py --long     # Slow tests only
python run_tests.py --not-long # All except slow tests
```

## Tips for Writing Tests

1. **Start with basic import tests** to ensure modules load correctly
2. **Test main functions** with simple input/output validation
3. **Use mocks** for expensive operations or external dependencies
4. **Add integration tests** for complex workflows
5. **Test error conditions** and edge cases

Example test structure:
```python
import pytest
import numpy as np
from vbi.inference import SomeClass, some_function

class TestSomeClass:
    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = np.random.randn(100, 10)
    
    def test_initialization(self):
        """Test object creation."""
        obj = SomeClass()
        assert obj is not None
    
    @pytest.mark.short
    def test_some_function_basic(self):
        """Test basic functionality."""
        result = some_function(self.test_data)
        assert result is not None
        assert result.shape[0] == self.test_data.shape[0]
    
    @pytest.mark.long
    def test_some_function_comprehensive(self):
        """Test with large dataset."""
        large_data = np.random.randn(10000, 100)
        result = some_function(large_data)
        # Add comprehensive assertions
```

## Next Steps

Priority modules to add tests for:
1. `vbi/inference.py` - Core inference functionality
2. `vbi/utils.py` - Utility functions
3. `feature_extraction/calc_features.py` - Feature calculation
4. `vbi/sbi_inference.py` - SBI-related inference
5. Model implementations (start with most used ones)

## Integration with CI/CD

Add to your CI pipeline:
```yaml
- name: Run tests with coverage
  run: |
    cd vbi/tests
    python run_tests.py --coverage --not-long
    
- name: Check coverage requirements
  run: |
    cd vbi/tests  
    python quick_coverage.py --missing
```
