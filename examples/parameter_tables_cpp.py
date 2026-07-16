"""Print parameter tables for all C++ models.

This module mirrors `parameter_tables_numba.py`: it imports selected C++
model modules, discovers `BaseModel` subclasses, instantiates each model
with minimal inputs, and prints the formatted parameter table.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np

from vbi.models.cpp.base import BaseModel

MODULES = [
    "vbi.models.cpp.vep",
    "vbi.models.cpp.jansen_rit",
    "vbi.models.cpp.mpr",
    "vbi.models.cpp.wc",
    "vbi.models.cpp.km",
    "vbi.models.cpp.damp_oscillator",
]


def _default_par(cls: type[BaseModel], nn: int = 2) -> dict:
    if cls.__name__ == "DO":
        return {}

    par = {"weights": np.eye(nn)}
    if cls.__name__ == "JR_sdde":
        par["delays"] = np.ones((nn, nn))
    if cls.__name__ == "KM_sde":
        par["omega"] = np.ones(nn)
    return par


def instantiate_and_print(module_name: str, nn: int = 2) -> None:
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"-- Failed to import {module_name}: {e}")
        return

    classes = [
        obj
        for _, obj in inspect.getmembers(mod, inspect.isclass)
        if issubclass(obj, BaseModel) and obj is not BaseModel
    ]
    if not classes:
        print(f"-- No BaseModel subclasses found in {module_name}")
        return

    for cls in classes:
        print("\n" + "#" * 80)
        print(f"Model: {cls.__name__} (from {module_name})")
        print("#" * 80)
        try:
            print(cls(_default_par(cls, nn)))
        except Exception as e:
            print(f"Failed to instantiate/print {cls.__name__}: {e}")


def main() -> None:
    for module_name in MODULES:
        instantiate_and_print(module_name)


if __name__ == "__main__":
    main()
