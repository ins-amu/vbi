"""
Simulator Models - Equations Reference
=======================================

Shows every model available in ``vbi.simulator.models`` as the last
expression of its own cell, so both Jupyter and this rendered gallery page
typeset the equations with MathJax via the model's rich HTML representation.

Use this page as a quick reference to check whether a model matches the
equations and parameter conventions you expect before running simulations.
"""

# %%
# Setup
# -----

from __future__ import annotations

from pathlib import Path
import sys

sys.dont_write_bytecode = True

try:
    _SCRIPT_PATH = Path(__file__)
except NameError:
    # sphinx-gallery execs this file without setting __file__; it already
    # chdirs into the script's own directory first, so cwd is equivalent.
    _SCRIPT_PATH = Path.cwd() / "model_equations.py"

# Prefer the vbi living in this checkout over any other version already
# installed (e.g. a different vbi checkout installed editable elsewhere).
# Downloaded standalone copies of this script should `pip install vbi`
# instead - see the first notebook cell.
_repo_root = _SCRIPT_PATH.resolve().parents[2]
sys.path.insert(0, str(_repo_root))

from vbi.simulator.models import (
    # QIF / mean-field
    mpr, coombes_byrne_2d, gast_sd, gast_sf,
    # Cortical column
    jansen_rit, wilson_cowan, larter_breakspear,
    # Wong-Wang family
    reduced_wong_wang, wong_wang_exc_inh,
    # Network oscillators
    sup_hopf, ghb, sl,
    # Epilepsy
    vep,
    # General / reference
    generic_2d_oscillator, kuramoto, linear, damped_oscillator,
)

# %%
# QIF mean-field models
# ----------------------
# Exact mean-field reductions of Quadratic Integrate-and-Fire (QIF) networks
# via the Ott-Antonsen ansatz.

# %%
# MontbrioPopulationRate
mpr

# %%
# CoombesByrne2D
coombes_byrne_2d

# %%
# GastSchmidtKnosche - Synaptic Depression
gast_sd

# %%
# GastSchmidtKnosche - Spike-Frequency Adaptation
gast_sf

# %%
# Cortical column models
# -----------------------
# Population-level descriptions of cortical columns with excitatory and
# inhibitory neural populations.

# %%
# JansenRit
jansen_rit

# %%
# WilsonCowan
wilson_cowan

# %%
# LarterBreakspear
larter_breakspear

# %%
# Wong-Wang family
# ------------------
# Biophysically plausible mean-field models based on NMDA receptor dynamics.

# %%
# ReducedWongWang (1-population)
reduced_wong_wang

# %%
# ReducedWongWangExcInh (2-population)
wong_wang_exc_inh

# %%
# Network oscillator models
# ---------------------------
# Stuart-Landau / Hopf normal-form models. All three share the same
# mathematical structure but differ in coupling type and parameter
# conventions:
#
# - ``sup_hopf`` - linear coupling (``c_x``, ``c_y``); scalar ``a``, ``omega``
# - ``sl`` - Laplacian coupling (``c_x - G*row_sum*x``); scalar ``a``, ``omega``
# - ``ghb`` - Laplacian coupling (``c_x - G*row_sum*x``); per-node ``eta``, ``omega``

# %%
# SupHopf (linear coupling)
sup_hopf

# %%
# StuartLandau - scalar parameters, Laplacian coupling
sl

# %%
# GHB - per-node parameters, Laplacian coupling
ghb

# %%
# Epilepsy models
# -----------------
# Whole-brain seizure propagation models.

# %%
# VEP (Virtual Epileptic Patient - 2D seizure-permittivity)
vep

# %%
# General-purpose / reference models
# ------------------------------------

# %%
# Generic2dOscillator
generic_2d_oscillator

# %%
# Kuramoto
kuramoto

# %%
# Linear
linear

# %%
# DampedOscillator
damped_oscillator
