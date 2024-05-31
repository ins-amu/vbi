#!/usr/bin/bash

CXX=g++
CXXFLAGS  = -std=c++11 -fopenmp -O2  -fPIC -c
SWIGFLAGS = -c++ -python -shadow

# location of the Python header files
PYTHON_VERSION = 3.8
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
#PYTHON_INCLUDE = /p/software/jusuf/stages/2022/software/Python/3.9.6-GCCcore-11.2.0/include/python3.9

all: _do.so _jr_sde.so _jr_sdde.so _mpr_sde.so _sl_sdde.so _km_sde.so _wc_ode.so _ww_sde.so _vep.so

help:
	@echo "Usage:"
	@echo "  make all     - Compile all models (default)"
	@echo "  make clean   - Remove all .o, .so, .pyc, and SWIG generated files"
	@echo "  make <model> - Compile specific model, where <model> is one of:"
	@echo "                 do, jr_sde, jr_sdde, mpr_sde, sl_sdde, km_sde, wc_ode, ww_sde, vep"


# Model-specific targets
vep: _vep.so
km_sde: _km_sde.so
jr_sde: _jr_sde.so
jr_sdde: _jr_sdde.so
mpr_sde: _mpr_sde.so
sl_sdde: _sl_sdde.so
wc_ode: _wc_ode.so
ww_sde: _ww_sde.so
do: _do.so

# Pattern rule for compiling SWIG wrapped models
%_wrap.o : %_wrap.cpp %.hpp utility.hpp bold.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

# Pattern rule for SWIG interface files
%_wrap.cpp : %.i
	swig $(SWIGFLAGS) -o $@ $<

# Pattern rule for shared object files
_%.so: %_wrap.o
	$(CXX) -shared -fopenmp $< -o $@

# Phony targets
.PHONY: help clean all

clean:
	rm -rf *.o *.so *.pyc *_wrap.cpp
	rm -rf __pycache__
	rm -rf *.py
