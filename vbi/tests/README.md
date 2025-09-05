# VBI Test Categories

This document describes the test categorization system for VBI tests based on execution time.

## Overview

Tests are categorized into two main groups:
- **Short/Fast tests**: Quick unit tests that complete in under 1 second
- **Long/Slow tests**: Integration tests or computationally intensive tests that take 2+ seconds

## Test Categories

### Short/Fast Tests (`@pytest.mark.short`, `@pytest.mark.fast`)
- Feature extraction tests (`test_features.py`)
- Quick validation tests
- Most unit tests
- **Characteristics**: Fast feedback, suitable for development workflow

### Long/Slow Tests (`@pytest.mark.long`, `@pytest.mark.slow`)
- Model simulation tests (e.g., `test_mpr_numba.py`, `test_mpr_cupy.py`)
- Tests that involve actual neural network simulations
- **Characteristics**: More thorough but time-intensive

## Usage

### Using the Test Runner Script

```bash
cd vbi/tests/

# Run only fast tests (recommended for development)
python run_tests.py --short

# Run only slow tests 
python run_tests.py --long

# Run all tests except slow ones (good for CI)
python run_tests.py --not-long

# Run all tests (default)
python run_tests.py --all
```

### Using pytest directly

```bash
# Fast tests only
pytest . -m short

# Slow tests only  
pytest . -m long

# All except slow tests
pytest . -m "not long"

# All tests with timing information
pytest . --durations=10
```

### Important Note about Environment

For best results and to ensure no tests are skipped due to missing dependencies:

1. **Activate the `vbidevelop` conda environment** before running tests:
   ```bash
   conda activate vbidevelop
   ```

2. Then run your tests using any of the methods above.

## Files

- `run_tests.py` - Enhanced test runner with category support (permanent file)
- `test_suite.py` - Updated test suite with category support  
- `pytest.ini` - Configuration file with marker definitions

## Execution Time Comparison

Use `pytest --durations=10` to see current timing breakdown.

| Category | Use Case | Expected Performance |
|----------|----------|---------------------|
| Short only | Development, quick validation | Fastest feedback |
| Long only | Integration testing | Thorough validation |
| All tests | Full test suite, CI/CD | Complete coverage |
| Not long | CI/CD with time constraints | Good coverage, reasonable time |

## Adding New Tests

When adding new tests, categorize them based on expected execution time:

1. **Fast tests** (< 1 second): Add `@pytest.mark.short` and `@pytest.mark.fast` markers
   - Unit tests, simple calculations, mocking-based tests
   
2. **Slow tests** (> 2 seconds): Add `@pytest.mark.long` and `@pytest.mark.slow` markers  
   - Integration tests, simulations, file I/O heavy tests

3. **Monitor execution times**: Use `pytest --durations=10` to identify tests that should be recategorized

## Benefits

- **Faster development cycle**: Run quick tests during development
- **Efficient CI/CD**: Run appropriate test subset based on context  
- **Better resource management**: Choose test intensity based on available time
- **Clear expectations**: Developers know what to expect from each category
