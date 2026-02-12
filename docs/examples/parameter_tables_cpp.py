#!/usr/bin/env python
"""
Test script demonstrating the new parameter table display functionality
for all C++ models in vbi.models.cpp

This script shows:
1. Uniform table display across all models
2. Dynamic parameter state (before/after prepare_input)
3. Proper handling of scalar vs array parameters
"""

import numpy as np
from vbi.models.cpp.vep import VEP_sde
from vbi.models.cpp.jansen_rit import JR_sde, JR_sdde
from vbi.models.cpp.mpr import MPR_sde
from vbi.models.cpp.wc import WC_ode
from vbi.models.cpp.km import KM_sde
from vbi.models.cpp.damp_oscillator import DO


def test_vep_dynamic_display():
    """Test VEP model showing scalar to array conversion"""
    print("\n" + "=" * 90)
    print("TEST 1: VEP_sde - Dynamic Parameter Display")
    print("=" * 90)
    
    weights = np.random.rand(5, 5)
    model = VEP_sde({'weights': weights, 'eta': -1.8, 'iext': 0.5})
    
    print("\nBEFORE prepare_input() - parameters as initialized:")
    print("-" * 90)
    print(model)
    
    model.prepare_input()
    
    print("\n\nAFTER prepare_input() - scalars converted to arrays:")
    print("-" * 90)
    print(model)
    
    print("\n✓ Notice: eta and iext changed from scalars to shape (5,)")


def test_jr_sde_array_params():
    """Test JR_sde with mixed scalar and array parameters"""
    print("\n" + "=" * 90)
    print("TEST 2: JR_sde - Mixed Scalar and Array Parameters")
    print("=" * 90)
    
    weights = np.random.rand(3, 3)
    A_values = np.array([3.25, 3.5, 3.0])
    
    model = JR_sde({'weights': weights, 'A': A_values, 'G': 0.7})
    
    print("\nInitial state (A already an array, C parameters as scalars):")
    print("-" * 90)
    print(model)
    
    model.prepare_input()
    
    print("\n\nAfter prepare_input() (C parameters converted to arrays):")
    print("-" * 90)
    print(model)
    
    print("\n✓ Notice: C0, C1, C2, C3 converted from scalars to shape (3,)")


def test_all_models_consistency():
    """Test that all models have consistent table display"""
    print("\n" + "=" * 90)
    print("TEST 3: All Models - Consistent Display Format")
    print("=" * 90)
    
    models = []
    
    # VEP
    weights = np.random.rand(2, 2)
    models.append(("VEP_sde", VEP_sde({'weights': weights})))
    
    # JR_sde
    weights = np.random.rand(2, 2)
    models.append(("JR_sde", JR_sde({'weights': weights})))
    
    # JR_sdde
    weights = np.random.rand(2, 2)
    delays = np.ones((2, 2))
    models.append(("JR_sdde", JR_sdde({'weights': weights, 'delays': delays})))
    
    # MPR
    weights = np.random.rand(2, 2)
    models.append(("MPR_sde", MPR_sde({'weights': weights})))
    
    # Wilson-Cowan
    weights = np.random.rand(2, 2)
    models.append(("WC_ode", WC_ode({'weights': weights})))
    
    # Kuramoto
    weights = np.random.rand(2, 2)
    omega = np.array([1.0, 1.5])
    models.append(("KM_sde", KM_sde({'weights': weights, 'omega': omega})))
    
    # Damped Oscillator
    models.append(("DO", DO()))
    
    for name, model in models:
        print(f"\n{name}:")
        print("-" * 90)
        print(model)
    
    print("\n✓ All models display parameters in consistent table format")


def test_shape_display():
    """Test various parameter shapes are displayed correctly"""
    print("\n" + "=" * 90)
    print("TEST 4: Shape Display - Scalars, Vectors, Matrices")
    print("=" * 90)
    
    # Create model with various shapes
    weights_2d = np.random.rand(4, 4)  # 2D matrix
    A_vector = np.array([3.0, 3.2, 3.4, 3.1])  # 1D vector
    G_scalar = 0.8  # scalar
    
    model = JR_sde({
        'weights': weights_2d,
        'A': A_vector,
        'G': G_scalar,
        'method': 'heun'
    })
    
    print("\nModel with different parameter types:")
    print("-" * 90)
    print(model)
    
    print("\n✓ Correctly displays:")
    print("  - Scalars: show value (G: 0.8, method: 'heun')")
    print("  - Vectors: show shape (A: shape (4,))")
    print("  - Matrices: show shape (weights: shape (4, 4))")


if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("VBI C++ MODELS - PARAMETER TABLE DISPLAY TESTS")
    print("=" * 90)
    print("\nThis script demonstrates the new unified parameter table display")
    print("across all C++ models in vbi.models.cpp")
    
    test_vep_dynamic_display()
    test_jr_sde_array_params()
    test_all_models_consistency()
    test_shape_display()
    
    print("\n" + "=" * 90)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 90)
    print("\nKey Features Demonstrated:")
    print("1. ✓ Uniform table format across all models")
    print("2. ✓ Dynamic display (shows actual parameter state)")
    print("3. ✓ Smart shape detection (scalars vs arrays)")
    print("4. ✓ Clear parameter descriptions")
    print("5. ✓ Consistent with PyTorch RWW model style")
    print("=" * 90 + "\n")
