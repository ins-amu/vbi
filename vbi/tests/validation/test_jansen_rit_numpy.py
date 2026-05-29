"""
M0 validation: NumPy backend - deterministic Jansen-Rit.
"""

import numpy as np
import pytest

from vbi.simulator import Simulator
from vbi.simulator.models.jansen_rit import jansen_rit
from vbi.simulator.spec import (
    CouplingSpec,
    IntegratorSpec,
    MonitorSpec,
    SimulationSpec,
)


JR_PARAMS = {
    "A": 3.25,
    "B": 22.0,
    "a": 0.1,
    "b": 0.05,
    "v0": 5.52,
    "nu_max": 0.0025,
    "r": 0.56,
    "J": 135.0,
    "a_1": 1.0,
    "a_2": 0.8,
    "a_3": 0.25,
    "a_4": 0.25,
    "mu": 0.22,
}


def make_jr_spec(
    weights: np.ndarray,
    dt: float = 0.01,
    method: str = "heun",
    coupling_strength: float = 0.5,
    tract_lengths: np.ndarray | None = None,
) -> SimulationSpec:
    if tract_lengths is None:
        tract_lengths = np.zeros_like(weights)
    return SimulationSpec(
        model=jansen_rit,
        integrator=IntegratorSpec(method=method, dt=dt),
        coupling=CouplingSpec("linear", a=coupling_strength),
        monitors=(MonitorSpec("raw"),),
        weights=weights,
        tract_lengths=tract_lengths,
        speed=1.0,
        node_params={
            name: np.full(weights.shape[0], value)
            for name, value in JR_PARAMS.items()
        },
    )


def run_tvb_jr_reference(
    weights: np.ndarray,
    dt: float,
    method: str,
    duration: float,
    coupling_strength: float,
    tract_lengths: np.ndarray | None = None,
) -> np.ndarray:
    pytest.importorskip("tvb")

    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator.coupling import Linear
    from tvb.simulator.integrators import EulerDeterministic, HeunDeterministic
    from tvb.simulator.models.jansen_rit import JansenRit
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

    tvb_model = JansenRit(
        A=np.array([JR_PARAMS["A"]]),
        B=np.array([JR_PARAMS["B"]]),
        a=np.array([JR_PARAMS["a"]]),
        b=np.array([JR_PARAMS["b"]]),
        v0=np.array([JR_PARAMS["v0"]]),
        nu_max=np.array([JR_PARAMS["nu_max"]]),
        r=np.array([JR_PARAMS["r"]]),
        J=np.array([JR_PARAMS["J"]]),
        a_1=np.array([JR_PARAMS["a_1"]]),
        a_2=np.array([JR_PARAMS["a_2"]]),
        a_3=np.array([JR_PARAMS["a_3"]]),
        a_4=np.array([JR_PARAMS["a_4"]]),
        mu=np.array([JR_PARAMS["mu"]]),
        variables_of_interest=("y0", "y1", "y2", "y3", "y4", "y5"),
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


class TestJansenRitNumpy:
    def test_two_nodes_run(self):
        weights = np.array([[0.0, 0.2], [0.3, 0.0]])
        spec = make_jr_spec(weights, coupling_strength=0.2)
        t, data = Simulator(spec, backend="numpy").run(10.0)["raw"]
        assert data.shape == (t.shape[0], 6, 2)
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
        coupling_strength = 0.4

        spec = make_jr_spec(weights, dt=dt, method=method,
                            coupling_strength=coupling_strength)
        _times, vbi_data = Simulator(spec, backend="numpy").run(duration)["raw"]
        tvb_data = run_tvb_jr_reference(
            weights, dt, method, duration, coupling_strength
        )

        assert vbi_data.shape == tvb_data.shape
        np.testing.assert_allclose(vbi_data, tvb_data, rtol=2e-10, atol=2e-12)
