"""Shared pytest configuration for the vbi test suite."""

from __future__ import annotations

import os


# Many tests import ``jax`` directly at module import time, before the inference
# backend registry can apply its CPU default.  Default to CPU in tests so
# machines with a visible but unusable CUDA/cuDNN setup do not fail during XLA
# compilation.  Developers can still override this explicitly, for example:
# ``JAX_PLATFORMS=gpu python -m pytest ...``.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
