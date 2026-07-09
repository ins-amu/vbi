"""Tests for the JIDT (infodynamics.jar) subprocess bridge.

JIDT is GPL-3.0-licensed; VBI runs it in an independent OS process
(``vbi.feature_extraction.jidt_worker``) rather than embedding its JVM
in-process, so that VBI's own Apache-2.0 process never links against
GPL-3.0 code. See ``docs/third_party_licenses.rst`` for the rationale.

These tests both check numerical correctness of the bridged calculators
and verify the process-isolation property itself.
"""
import importlib.util
import os
import unittest

import numpy as np
import pytest

from vbi.feature_extraction import features_utils
from vbi.feature_extraction.features import calc_mi, calc_te, calc_entropy

_HAS_JPYPE = importlib.util.find_spec("jpype") is not None


def _coupled_series(n=2000, seed=0):
    """AR(1)-coupled pair: y depends on a lagged copy of x, so MI/TE > 0."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.9 * x[t - 1] + 0.1 * rng.standard_normal()
    return np.vstack([x, y])


@pytest.mark.long
@pytest.mark.skipif(not _HAS_JPYPE, reason="JPype not installed")
class TestJidtBridgeCorrectness(unittest.TestCase):
    def setUp(self):
        features_utils._shutdown_jidt_process()

    def tearDown(self):
        features_utils._shutdown_jidt_process()

    def test_calc_mi_detects_coupling(self):
        ts = _coupled_series(seed=1)
        values, labels = calc_mi(ts, k=4, source_indices=[0], target_indices=[1], mode="pairwise")
        self.assertEqual(labels, ["mi"])
        self.assertGreater(values[0], 0.1)

    def test_calc_te_detects_directed_coupling(self):
        ts = _coupled_series(seed=2)
        te_xy, _ = calc_te(ts, k=4, source_indices=[0], target_indices=[1], mode="pairwise")
        te_yx, _ = calc_te(ts, k=4, source_indices=[1], target_indices=[0], mode="pairwise")
        self.assertGreater(te_xy[0], te_yx[0])

    def test_calc_entropy_per_region_and_average(self):
        ts = _coupled_series(seed=3)
        values, labels = calc_entropy(ts, average=False)
        self.assertEqual(len(values), 2)
        self.assertTrue(all(np.isfinite(v) for v in values))
        self.assertEqual(labels, ["entropy_0", "entropy_1"])

        avg_value, avg_label = calc_entropy(ts, average=True)
        self.assertTrue(np.isfinite(avg_value))
        self.assertEqual(avg_label, "entropy")


@pytest.mark.long
@pytest.mark.skipif(not _HAS_JPYPE, reason="JPype not installed")
class TestJidtProcessIsolation(unittest.TestCase):
    """Verify JIDT actually runs out-of-process, not embedded via in-process JPype."""

    def setUp(self):
        features_utils._shutdown_jidt_process()

    def tearDown(self):
        features_utils._shutdown_jidt_process()

    def test_jidt_runs_in_a_separate_process(self):
        self.assertIsNone(features_utils._jidt_process)
        features_utils.call_jidt("entropy", ts=[[0.1, 0.2, 0.3, 0.4, 0.5]], average=True)

        proc = features_utils._jidt_process
        self.assertIsNotNone(proc)
        self.assertNotEqual(proc.pid, os.getpid())
        self.assertIsNone(proc.poll(), "worker process should still be alive")

        # The GPL-relevant fact isn't whether the (BSD-licensed) jpype
        # Python package is imported here -- it's that the JVM, and
        # therefore infodynamics.jar itself, never starts in this
        # process. Only the worker subprocess is allowed to start it.
        import jpype as jp
        self.assertFalse(jp.isJVMStarted())

    def test_shutdown_terminates_worker_process(self):
        features_utils.call_jidt("entropy", ts=[[0.1, 0.2, 0.3, 0.4, 0.5]], average=True)
        proc = features_utils._jidt_process
        self.assertIsNone(proc.poll())

        features_utils._shutdown_jidt_process()

        self.assertIsNone(features_utils._jidt_process)
        proc.wait(timeout=5)
        self.assertIsNotNone(proc.poll(), "worker process should have exited")

    def test_worker_process_reused_across_calls(self):
        features_utils.call_jidt("entropy", ts=[[0.1, 0.2, 0.3]], average=True)
        first_pid = features_utils._jidt_process.pid
        features_utils.call_jidt("entropy", ts=[[0.4, 0.5, 0.6]], average=True)
        second_pid = features_utils._jidt_process.pid
        self.assertEqual(first_pid, second_pid)
