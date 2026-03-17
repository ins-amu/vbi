# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.3] - 2026-02-17

### Added
- Unified parameter API tests for CuPy models.
- New tests for the Reduced Wong-Wang (RWW) model, including parameter access, validation, and BOLD signal generation.
- A notebook for plotting examples with 1D and 2D distributions.
- Parameter description methods across multiple C++ models.
- Improved parameter table display test coverage for C++ models.
- API docs now include automodule directives for base models in CuPy and Numba sections.
- GitHub Actions workflow to auto-update `codemeta.json`.

### Changed
- Refactored CuPy models to inherit from `BaseCupyModel` with unified parameter management and descriptions (GHB, JR, KM, MPR, WC, WW).
- Refactored parameter handling in C++ models (`JR_sdde`, `KM_sde`, `MPR_sde`, `VEP_sde`, `WC_ode`) for more consistent display and access.
- Refactored parameter handling and compatibility naming for Numba models (`JR_sde`, `VEP_sde`, `WC_ode`).
- Refactored Stuart-Landau model parameter access for consistency.
- Updated docs theme to Furo and reorganized parts of inference documentation.
- Updated installation/workflow documentation around optional C++ compilation steps and dependencies.
- Refactored CuPy model imports and version update logic for metadata maintenance.
- Updated version to 0.4.3.

### Fixed
- Fixed notebook markdown header formatting for Jansen-Rit whole-brain (NUMBA) examples.
- Improved error handling for C++ module imports in Jansen-Rit and VEP tests.
- Minor code cleanup in VEP (`get_parameter_descriptions` removal) and formatting/clarity improvements in MPR outputs.

## [0.4.1] - 2026-02-06

### Added
- **New Models:**
  - JAX support for parallel simulations with `vmap` and `pmap` for multi-core CPU execution
  - MPR model wrapper for TVBK simulation framework
  - BOLD model parameters and integration step function enhancements
  - Numba implementation of Stuart-Landau oscillator model with comprehensive tests
  - Reduced Wong-Wang (RWW) model implementation with BOLD recording capabilities
  - Reduced Wong-Wang Mean Field Model with hemodynamic response in PyTorch
  - Stuart-Landau whole-brain model description and documentation

- **Feature Enhancements:**
  - Enhanced parallel simulation capabilities for parameter sweeps
  - Comprehensive tests for fc_correlation_cost and fcd_ks_cost functions with PyTorch implementation
  - LOAD_DATA variable in notebooks to avoid recomputation when simulation data is available
  - Enhanced API documentation and quick start guides for CDE and MDN
  - Improved error handling for torch imports with skip functionality in tests

### Changed
- Updated version to 0.4.1
- Improved simulation performance with JAX parallelization options
- Updated project homepage and repository URLs in pyproject.toml
- Enhanced documentation for Reduced Wong-Wang model implementation
- Updated download URL and release notes for version 0.4.1 in codemeta.json
- Updated pip install command to include inference extras

### Fixed
- Updated version retrieval in Sphinx configuration to use _version.py
- Corrected example paths for RWW models in documentation
- Removed unused imports of torch and math in utils.py
- Updated file paths in ww_full_sde_cupy notebook for consistency
- Fixed notebook importing issues (ww_sde_kong -> rww_sde_kong)
- Resolved loading simulation data issues in notebooks
- Updated vbi_log.png to reflect recent changes

### Removed
- Removed unused imports and cleaned up code structure

## [0.3] - 2025-09-17

### Added
- Conditional Density Estimator (CDE) module with MDN and MAF implementations
- MAFEstimator0 class for enhanced conditional density estimation with autograd dependency
- NumPy-based implementations for posterior shrinkage, z-score calculations, and pairplot
- New example notebooks for CDE with MAF and MDN, including VEP, damp oscillator, and toy examples
- Tutorials for CDE inference
- Test suite for posterior shrinkage and z-score functions
- 'corner' library to project dependencies for visualization
- GDPR compliance statement and reference in README
- Executed_notebooks to .gitignore
- Parameter validation and utility functions for model consistency
- Windows installation instructions and improved Catch22 feature handling
- Detailed installation guide for Google Colab and EBRAINS

### Changed
- Enhanced posterior_peaks function to prefer sbi-based plotting with fallback to numpy
- Refactored conditional imports and error handling for optional dependencies
- Updated import paths for inference modules across notebooks
- Improved makefile warning suppression and added strict compilation targets
- Simplified Windows installation by automating C++ compilation detection
- Updated API documentation and added new models for Numba and PyTorch
- Enhanced documentation for various models and functions
- Updated Docker build configuration and added 'all-docker' dependencies

### Fixed
- Added export command to skip C++ compilation in installation instructions
- Updated C++ compilation instructions and enhanced dependency checks
- Clarified C++ compilation settings in troubleshooting section
- Enhanced C++ build process with improved opt-in handling and error messaging
- Updated import path for posterior_peaks in tests
- Fixed file paths to use variable for output directory
- Raised ImportError with instructions if sbi package is not installed
- Cleaned up messy output cells in notebooks

### Removed
- Removed outdated installation guide from repository

## [0.2.2] - 2025-09-05

### Added
- Implemented Numba-based Virtual Epileptic Patient (VEP) model
- Added comprehensive Docker support with Dockerfile, .dockerignore, and startup script
- Added Docker build and quick start documentation
- Added example notebooks for Jansen-Rit and Wilson-Cowan models with Numba support
- Enhanced test suite with pytest markers for short/fast and long/slow tests
- Implemented test categorization and enhanced test runner
- Added informative error handling for missing PyTorch and SBI dependencies

### Changed
- Updated installation instructions in documentation for clarity and flexibility
- Enhanced optional dependency handling and installation instructions
- Updated VBI Docker script with confirmation prompts and auto-build functionality
- Updated Docker workflow for proper branch handling and authentication
- Bumped version to 0.2.2

### Fixed
- Removed unused imports and cleaned up code structure in VEP model
- Updated required Sphinx version to 0.2 in configuration
- Fixed heading formatting in quick start guide

## [0.2] - 2025-08-28

### Added
- **New Models:**
  - Implemented Wong-Wang model with Numba JIT compilation and simulation functionality
  - Added Numba-based Jansen-Rit model with initial state setup and integration functionality
  - Implemented Bold model with parameter management and BOLD step functionality
  - Added Wilson-Cowan CuPy implementation and example notebook
  - Added Wong-Wang full (EI) model with CuPy implementation

- **Feature Enhancements:**
  - Added `__str__` method to JR_sde class for improved model parameter representation
  - Added average parameter to `spectrum_moments` function for optional averaging of moments
  - Enhanced `integrate` function to filter recorded data based on `t_cut`, with assertion for validity
  - Added `dtype_convert` function and enhanced `prepare_vec_2d` for flexible input handling
  - Added optional parameter 'k' to `fcd_stat` function for flexible window length calculation
  - Added functions to check data location on CPU/GPU
  - Enhanced parameter handling in KM_sde class
  - Added SKIP_CPP environment variable to control C++ compilation during setup
  - Added error handling for PCA transformation in `matrix_stat` function
  - Allow customizable wavelet function in wavelet method (replaced ricker with morlet2)
  - Added `allocate_memory` method in Bold class to accept dtype parameter

- **Documentation:**
  - Updated installation instructions to include SKIP_CPP environment variable
  - Added Colab links and enhanced example notebooks with additional markdown headers
  - Updated installation instructions in README to include pip install options
  - Added docstring for train method in Inference class with parameter descriptions
  - Added new example directories for mpr_sde_cpp and wilson_cowan_cupy
  - Enhanced Jansen-Rit model documentation and improved CSS styles
  - Added custom CSS styles and updated documentation structure
  - Updated VBI toolkit description and improved presentation

- **Infrastructure:**
  - Added tvbk module integration with connectivity setup and MPR class implementation
  - Added CONTRIBUTING.md for contribution guidelines
  - Enhanced Docker support with NVIDIA CUDA base image and Python 3.10
  - Added GitHub Actions workflow for publishing to PyPI
  - Added Docker build badge and usage instructions

### Changed
- **Version Updates:**
  - Updated project version from 0.1.3 to 0.2 in both `_version.py` and `pyproject.toml`
  - Refactored DO_cpp class to DO

- **Code Improvements:**
  - Refactored Numba models and removed unused files
  - Improved code formatting and style in Inference class methods
  - Enhanced parameter handling in JR_sde and MPR_sde classes
  - Updated buffer size calculation and streamlined data recording in WC_sde class
  - Renamed `cupy/wilsoncowan.py` to `cupy/wilson_cowan.py` for naming compatibility

- **Build System:**
  - Updated MANIFEST.in and setup.py to include additional data files and C++ extensions
  - Cleaned up pyproject.toml by removing commented sections
  - Updated Docker workflow and improved build process

### Fixed
- Corrected weights array transposition in JR_sde class for proper data handling
- Removed unnecessary transpose operation on weights in MPR_sde class
- Updated noise_amp handling and added dependent parameter updates in MPR_sde class
- Restored dtype definition in MPR class initialization
- Updated import error messages to clarify C++ compilation or linking issues
- Extended r_period calculation and improved RV recording condition in MPR_sde class
- Fixed wavelet function imports and implementation

### Removed
- Removed tvbk_ext module and associated imports
- Removed pre-commit configuration
- Cleaned up deprecated test files and unnecessary dependencies
- Removed outdated publish workflow steps from GitHub Actions

## [0.1.3.3] - 2025-02-17

### Changed
- Updated package_data in setup.py to include Python files in vbi.models.cpp._src directory

## [0.1.3.2] - 2025-02-17

### Added
- Updated wavelet functions to allow None as default for customizable wavelet function

## [0.1.3.1] - 2025-02-17

### Added
- Imported models module in vbi package initialization
- Implemented Bold class for balloon model and integrated with MPR_sde class

### Changed
- Updated Python version in publish workflow to 3.10 for improved compatibility

## [0.1.3] - 2025-02-17

### Added
- Dynamic version retrieval implementation
- Enhanced documentation structure and API references

### Changed
- Bumped version to 0.1.3

---

**Note:** This changelog covers changes from version 0.1.3 to the current version 0.2. Earlier versions may have additional changes not documented here.
