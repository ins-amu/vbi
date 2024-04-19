#!/usr/bin/bash

CXX=g++
CXXFLAGS  = -std=c++11 -fopenmp -O2  -fPIC -c
SWIGFLAGS = -c++ -python -shadow

# location of the Python header files
PYTHON_VERSION = 3.8
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
#PYTHON_INCLUDE = /p/software/jusuf/stages/2022/software/Python/3.9.6-GCCcore-11.2.0/include/python3.9

all: _do.so _jr_sde.so _jr_sdde.so _mpr_sde.so _sl_sdde.so _km_sde.so _wc_ode.so _ww_sde.so

help: # add some explanation
	@echo "make all     (default)"
	@echo "make clean   (remove all .o, .so, .pyc, and swig generated files)"
	@echo "make do      (compile Damping Oscillator model)"
	@echo "make jr_sde  (compile Jansen-Rit SDE model without delay)"
	@echo "make jr_sdde (compile Jansen-Rit SDDE model with delay and noise)"
	@echo "make mpr_sde (compile Montbrio-Pazo-Roxin SDE model)"
	@echo "make sl_sdde (compile Stuart-Landau SDDE model)"
	@echo "make km_sde  (compile Kuramoto-Sakaguchi SDE model)"
	@echo "make wc_ode  (compile Wilson-Cowan ODE model)"
	@echo "make ww_sde  (compile Wong-Wang SDE model)"


km_sde: _km_sde.so

jr_sde: _jr_sde.so

jr_sdde: _jr_sdde.so

mpr_sde: _mpr_sde.so

sl_sdde: _sl_sdde.so

wc_ode: _wc_ode.so

ww_sde: _ww_sde.so

do: _do.so

# -----------------------------------------------------------------------------
_do.so: do_wrap.o
	$(CXX) -shared -fopenmp do_wrap.o -o _do.so

do_wrap.o : do_wrap.cpp do.hpp utility.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

do_wrap.cpp : do.i
	swig $(SWIGFLAGS) -o do_wrap.cpp do.i

# -----------------------------------------------------------------------------
_jr_sde.so: jr_sde_wrap.o
	$(CXX) -shared -fopenmp jr_sde_wrap.o -o _jr_sde.so

jr_sde_wrap.o : jr_sde_wrap.cpp jr_sde.hpp utility.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

jr_sde_wrap.cpp : jr_sde.i
	swig $(SWIGFLAGS) -o jr_sde_wrap.cpp jr_sde.i
# -----------------------------------------------------------------------------

_jr_sdde.so: jr_sdde_wrap.o
	$(CXX) -shared -fopenmp jr_sdde_wrap.o -o _jr_sdde.so

jr_sdde_wrap.o : jr_sdde_wrap.cpp jr_sdde.hpp utility.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

jr_sdde_wrap.cpp : jr_sdde.i
	swig $(SWIGFLAGS) -o jr_sdde_wrap.cpp jr_sdde.i
# -----------------------------------------------------------------------------

_mpr_sde.so: mpr_sde_wrap.o
	$(CXX) -shared -fopenmp mpr_sde_wrap.o -o _mpr_sde.so

mpr_sde_wrap.o : mpr_sde_wrap.cpp mpr_sde.hpp utility.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

mpr_sde_wrap.cpp : mpr_sde.i
	swig $(SWIGFLAGS) -o mpr_sde_wrap.cpp mpr_sde.i
# -----------------------------------------------------------------------------

_sl_sdde.so: sl_sdde_wrap.o
	$(CXX) -shared -fopenmp sl_sdde_wrap.o -o _sl_sdde.so

sl_sdde_wrap.o : sl_sdde_wrap.cpp sl_sdde.hpp utility.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

sl_sdde_wrap.cpp : sl_sdde.i
	swig $(SWIGFLAGS) -o sl_sdde_wrap.cpp sl_sdde.i
# -----------------------------------------------------------------------------

_km_sde.so: km_sde_wrap.o
	$(CXX) -shared -fopenmp km_sde_wrap.o -o _km_sde.so

km_sde_wrap.o : km_sde_wrap.cpp km_sde.hpp utility.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

km_sde_wrap.cpp : km_sde.i
	swig $(SWIGFLAGS) -o km_sde_wrap.cpp km_sde.i
# -----------------------------------------------------------------------------

_wc_ode.so: wc_ode_wrap.o
	$(CXX) -shared -fopenmp wc_ode_wrap.o -o _wc_ode.so

wc_ode_wrap.o : wc_ode_wrap.cpp wc_ode.hpp utility.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

wc_ode_wrap.cpp : wc_ode.i
	swig $(SWIGFLAGS) -o wc_ode_wrap.cpp wc_ode.i
# -----------------------------------------------------------------------------

_ww_sde.so: ww_sde_wrap.o
	$(CXX) -shared -fopenmp ww_sde_wrap.o -o _ww_sde.so

ww_sde_wrap.o : ww_sde_wrap.cpp ww_sde.hpp utility.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

ww_sde_wrap.cpp : ww_sde.i
	swig $(SWIGFLAGS) -o ww_sde_wrap.cpp ww_sde.i
# -----------------------------------------------------------------------------

.PHONEY: help clean all km_sde jr_sde jr_sdde mpr_sde sl_sdde do

clean :
	rm -rf *.o  *.so *.pyc
	rm -rf do_wrap.cpp do.py
	rm -rf jr_sde_wrap.cpp jr_sde.py
	rm -rf jr_sdde_wrap.cpp jr_sdde.py
	rm -rf mpr_sde_wrap.cpp mpr_sde.py
	rm -rf sl_sdde_wrap.cpp sl_sdde.py
	rm -rf km_sde_wrap.cpp km_sde.py
	rm -rf wc_ode_wrap.cpp wc_ode.py
	rm -rf ww_sde_wrap.cpp ww_sde.py
	rm -rf __pycache__
