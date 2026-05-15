"""
M0 validation: NumPy backend — deterministic Wilson-Cowan.
"""

import numpy as np
import pytest

from vbi.simulator import Simulator
from vbi.simulator.models.wilson_cowan import wilson_cowan
from vbi.simulator.spec import (
    CouplingSpec,
    IntegratorSpec,
    MonitorSpec,
    SimulationSpec,
)


WC_PARAMS = {
    "c_ee": 12.0,
    "c_ei": 4.0,
    "c_ie": 13.0,
    "c_ii": 11.0,
    "tau_e": 10.0,
    "tau_i": 10.0,
    "a_e": 1.2,
    "b_e": 2.8,
    "c_e": 1.0,
    "theta_e": 0.0,
    "a_i": 1.0,
    "b_i": 4.0,
    "c_i": 1.0,
    "theta_i": 0.0,
    "r_e": 1.0,
    "r_i": 1.0,
    "k_e": 1.0,
    "k_i": 1.0,
    "P": 0.5,
    "Q": 0.0,
    "alpha_e": 1.0,
    "alpha_i": 1.0,
}


def make_wc_spec(
    weights: np.ndarray,
    dt: float = 0.01,
    method: str = "heun",
    coupling_strength: float = 0.15,
    tract_lengths: np.ndarray | None = None,
) -> SimulationSpec:
    if tract_lengths is None:
        tract_lengths = np.zeros_like(weights)
    return SimulationSpec(
        model=wilson_cowan,
        integrator=IntegratorSpec(method=method, dt=dt),
        coupling=CouplingSpec("linear", a=coupling_strength),
        monitors=(MonitorSpec("raw"),),
        weights=weights,
        tract_lengths=tract_lengths,
        speed=1.0,
        node_params={
            name: np.full(weights.shape[0], value)
            for name, value in WC_PARAMS.items()
        },
    )


def run_tvb_wc_reference(
    weights: np.ndarray,
    dt: float,
    method: str,
    duration: float,
    coupling_strength: float,
    tract_lengths: np.ndarray | None = None,
) -> np.ndarray:
    tvb = pytest.importorskip("tvb")
    del tvb

    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator.coupling import Linear
    from tvb.simulator.integrators import EulerDeterministic, HeunDeterministic
    from tvb.simulator.models.wilson_cowan import WilsonCowan
    from tvb.simulator.monitors import Raw
    from tvb.simulator.simulator import Simulator as TVBSimulator

    if tract_lengths is None:
        tract_lengths = np.zeros_like(weights)

    n_nodes = weights.shape[0]
    conn = Connectivity(
        weights=weights,
        tract_lengths=tract_lengths,
        region_labels=np.array([str(i) for i in range(n_nodes)]),
        centres=np.zeros((n_nodes, 3)),
        speed=np.array([1.0]),
    )
    conn.configure()

    tvb_model = WilsonCowan(
        c_ee=np.array([WC_PARAMS["c_ee"]]),
        c_ei=np.array([WC_PARAMS["c_ei"]]),
        c_ie=np.array([WC_PARAMS["c_ie"]]),
        c_ii=np.array([WC_PARAMS["c_ii"]]),
        tau_e=np.array([WC_PARAMS["tau_e"]]),
        tau_i=np.array([WC_PARAMS["tau_i"]]),
        a_e=np.array([WC_PARAMS["a_e"]]),
        b_e=np.array([WC_PARAMS["b_e"]]),
        c_e=np.array([WC_PARAMS["c_e"]]),
        theta_e=np.array([WC_PARAMS["theta_e"]]),
        a_i=np.array([WC_PARAMS["a_i"]]),
        b_i=np.array([WC_PARAMS["b_i"]]),
        c_i=np.array([WC_PARAMS["c_i"]]),
        theta_i=np.array([WC_PARAMS["theta_i"]]),
        r_e=np.array([WC_PARAMS["r_e"]]),
        r_i=np.array([WC_PARAMS["r_i"]]),
        k_e=np.array([WC_PARAMS["k_e"]]),
        k_i=np.array([WC_PARAMS["k_i"]]),
        P=np.array([WC_PARAMS["P"]]),
        Q=np.array([WC_PARAMS["Q"]]),
        alpha_e=np.array([WC_PARAMS["alpha_e"]]),
        alpha_i=np.array([WC_PARAMS["alpha_i"]]),
        shift_sigmoid=np.array([True]),
        variables_of_interest=("E", "I"),
    )
    integrator_cls = {
        "euler": EulerDeterministic,
        "heun": HeunDeterministic,
    }[method]
    sim = TVBSimulator(
        connectivity=conn,
        model=tvb_model,
        coupling=Linear(a=np.array([coupling_strength])),
        integrator=integrator_cls(dt=dt),
        monitors=[Raw()],
        simulation_length=duration,
    ).configure()

    initial_state = np.zeros((tvb_model.nvar, n_nodes, 1))
    sim.current_state[:] = initial_state
    sim.history.buffer[:] = initial_state[tvb_model.cvar][np.newaxis, ...]

    (_times, data), = sim.run()
    return data[:, :, :, 0]


class TestWilsonCowanNumpy:
    def test_two_nodes_run(self):
        weights = np.array([[0.0, 0.2], [0.3, 0.0]])
        spec = make_wc_spec(weights, coupling_strength=0.2)
        t, data = Simulator(spec, backend="numpy").run(10.0)["raw"]
        assert data.shape == (t.shape[0], 2, 2)
        assert np.isfinite(data).all()

    @pytest.mark.parametrize("method", ["euler", "heun"])
    def test_raw_trajectory_matches_tvb_without_delays(self, method):
        weights = np.array([
            [0.0, 0.25, 0.10],
            [0.50, 0.0, 0.20],
            [0.15, 0.35, 0.0],
        ])
        dt = 0.01
        duration = 1.0
        coupling_strength = 0.15

        spec = make_wc_spec(
            weights,
            dt=dt,
            method=method,
            coupling_strength=coupling_strength,
        )
        _times, vbi_data = Simulator(spec, backend="numpy").run(duration)["raw"]
        tvb_data = run_tvb_wc_reference(
            weights, dt, method, duration, coupling_strength
        )

        assert vbi_data.shape == tvb_data.shape
        np.testing.assert_allclose(vbi_data, tvb_data, rtol=2e-10, atol=2e-12)
