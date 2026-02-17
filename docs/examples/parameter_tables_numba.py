"""Print parameter tables for all Numba models.

This script discovers model classes in `vbi.models.numba`, instantiates
each model with a minimal connectivity matrix, and prints the formatted
parameter table using the unified BaseNumbaModel API.

It is intended for documentation and sanity-checking.
"""
import importlib
import inspect
import pkgutil
import numpy as np

from vbi.models.numba.base import BaseNumbaModel

MODULE_PREFIX = 'vbi.models.numba'

# List of modules to try (explicit to avoid importing heavy deps accidentally)
modules = [
    'vbi.models.numba.jansen_rit',
    'vbi.models.numba.vep',
    'vbi.models.numba.wilson_cowan',
    'vbi.models.numba.mpr',
    'vbi.models.numba.ww',
    'vbi.models.numba.rww',
    'vbi.models.numba.ghb',
    'vbi.models.numba.sl',
    'vbi.models.numba.damp_oscillator',
]


def instantiate_and_print(module_name):
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"-- Failed to import {module_name}: {e}")
        return

    # find classes in module that are subclasses of BaseNumbaModel
    classes = [
        obj for _, obj in inspect.getmembers(mod, inspect.isclass)
        if issubclass(obj, BaseNumbaModel) and obj is not BaseNumbaModel
    ]

    if not classes:
        print(f"-- No BaseNumbaModel subclasses found in {module_name}")
        return

    for cls in classes:
        print("\n" + "#" * 80)
        print(f"Model: {cls.__name__} (from {module_name})")
        print("#" * 80)
        # try to create with a small identity weights matrix (nn=2)
        try:
            inst = cls({'weights': np.eye(2)})
            print(inst)
        except Exception as e:
            print(f"Failed to instantiate/print {cls.__name__}: {e}")


if __name__ == '__main__':
    for m in modules:
        instantiate_and_print(m)
