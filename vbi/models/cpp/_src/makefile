# Makefile for VBI C++ models with SWIG Python bindings
# Improved version with better practices and features

# Compiler and flags
CXX ?= g++
PYTHON ?= python3

# Detect number of cores for parallel compilation
NPROC := $(shell nproc 2>/dev/null || echo 1)

# Build configuration
BUILD_TYPE ?= release
DEBUG_FLAGS = -g -O0 -DDEBUG
RELEASE_FLAGS = -O2 -DNDEBUG

# Base compiler flags
CXXFLAGS_BASE = -std=c++11 -fopenmp -fPIC -Wall -Wextra
# Suppress warnings that are common in SWIG-generated code but keep important ones
# Note: Consider fixing these in source files rather than suppressing
CXXFLAGS_WARNINGS = -Wno-unused-parameter -Wno-sign-compare
CXXFLAGS_BUILD = $(if $(filter debug,$(BUILD_TYPE)),$(DEBUG_FLAGS),$(RELEASE_FLAGS))
CXXFLAGS = $(CXXFLAGS_BASE) $(CXXFLAGS_WARNINGS) $(CXXFLAGS_BUILD) -c

# SWIG flags
SWIGFLAGS = -c++ -python -shadow -O

# Python configuration
PYTHON_INCLUDE := $(shell $(PYTHON)-config --includes 2>/dev/null || echo "-I/usr/include/python3")
PYTHON_LIBS := $(shell $(PYTHON)-config --ldflags 2>/dev/null || echo "")

# Model definitions
MODELS = do jr_sde jr_sdde mpr_sde km_sde wc_ode vep
SHARED_LIBS = $(addprefix _,$(addsuffix .so,$(MODELS)))
WRAP_OBJS = $(addsuffix _wrap.o,$(MODELS))
WRAP_CPPS = $(addsuffix _wrap.cpp,$(MODELS))

# Default target
all: $(SHARED_LIBS)

# Parallel build support
.PHONY: parallel
parallel:
	$(MAKE) -j$(NPROC) all

# Help target with improved information
help:
	@echo "VBI C++ Models Makefile"
	@echo "======================="
	@echo "Available targets:"
	@echo "  all          - Compile all models (default)"
	@echo "  parallel     - Compile all models using $(NPROC) parallel jobs"
	@echo "  clean        - Remove all build-generated files"
	@echo "  distclean    - Remove all generated files and simulation output"
	@echo "  install      - Install compiled modules (copies .so files)"
	@echo "  check        - Verify all required dependencies"
	@echo "  debug        - Build with debug flags (BUILD_TYPE=debug)"
	@echo "  strict       - Build with all warnings enabled (for development)"
	@echo ""
	@echo "Individual model targets:"
	@$(foreach model,$(MODELS),echo "  $(model)         - Compile $(model) model";)
	@echo ""
	@echo "Build configuration:"
	@echo "  BUILD_TYPE   = $(BUILD_TYPE) (debug/release)"
	@echo "  CXX          = $(CXX)"
	@echo "  PYTHON       = $(PYTHON)"
	@echo "  NPROC        = $(NPROC)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make                    # Build all models"
	@echo "  make parallel           # Build with parallel jobs"
	@echo "  make BUILD_TYPE=debug   # Build with debug info"
	@echo "  make jr_sde             # Build only jr_sde model"

# Individual model targets (automatically generated)
$(MODELS): %: _%.so

# Debug build target
debug:
	$(MAKE) BUILD_TYPE=debug all

# Strict build target for development (shows all warnings)
strict:
	$(MAKE) CXXFLAGS_WARNINGS="" all

# Installation target
install: all
	@echo "Installing compiled modules..."
	@mkdir -p ../
	@cp *.so ../ 2>/dev/null || true
	@cp *.py ../ 2>/dev/null || true
	@echo "Installation complete."

# Dependency check
check:
	@echo "Checking dependencies..."
	@which $(CXX) >/dev/null || (echo "Error: $(CXX) not found" && exit 1)
	@which swig >/dev/null || (echo "Error: SWIG not found" && exit 1)
	@$(PYTHON) -c "import sys; print(f'Python version: {sys.version}')" || (echo "Error: $(PYTHON) not working" && exit 1)
	@echo "All dependencies OK."

# Build rules with improved error handling and verbosity
# Pattern rule for compiling SWIG wrapped models
%_wrap.o: %_wrap.cpp %.hpp utility.hpp bold.hpp
	@echo "Compiling $@..."
	$(CXX) $(CXXFLAGS) $< $(PYTHON_INCLUDE) -o $@

# Pattern rule for SWIG interface files
%_wrap.cpp: %.i %.hpp
	@echo "Generating SWIG wrapper for $*..."
	swig $(SWIGFLAGS) -o $@ $<

# Pattern rule for shared object files
_%.so: %_wrap.o
	@echo "Linking $@..."
	$(CXX) -shared -fopenmp $< $(PYTHON_LIBS) -o $@

# Phony targets
.PHONY: help clean distclean all parallel debug strict install check $(MODELS)

# Enhanced clean targets
clean:
	@echo "Cleaning generated files..."
	@rm -f *.o *.so *.pyc $(WRAP_CPPS) *_wrap.cxx
	@rm -f $(addsuffix .py,$(MODELS))
	@rm -rf __pycache__
	@echo "Clean complete."

distclean: clean
	@echo "Deep cleaning..."
	@rm -rf output/
	@rm -f *.py~  *.hpp~ *.i~ makefile~
	@echo "Deep clean complete."

# Show build configuration
config:
	@echo "Build Configuration:"
	@echo "  CXX          = $(CXX)"
	@echo "  CXXFLAGS     = $(CXXFLAGS)"
	@echo "  SWIGFLAGS    = $(SWIGFLAGS)"
	@echo "  PYTHON       = $(PYTHON)"
	@echo "  PYTHON_INCLUDE = $(PYTHON_INCLUDE)"
	@echo "  PYTHON_LIBS  = $(PYTHON_LIBS)"
	@echo "  BUILD_TYPE   = $(BUILD_TYPE)"
	@echo "  MODELS       = $(MODELS)"

# Makefile debugging
print-%:
	@echo $* = $($*)
	