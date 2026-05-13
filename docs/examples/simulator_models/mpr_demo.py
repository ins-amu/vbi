"""Compare VBI simulator and TVB MPR trajectories.

The network and working-point parameters mirror
``docs/examples/mpr_sde_numba.ipynb``:

    nn = 6
    weights = complete_graph(nn)
    G = 0.33
    I = 2.0
    delay = 1.0 ms
    dt = 0.01 ms
    tau = 1.0
    eta = -4.6

The notebook also uses ``noise_amp`` for the Numba SDE model. This comparison is
deterministic so that TVB and the new simulator can be compared point by point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from helpers import (
    comparison_metrics,
    complete_graph_weights,
    constant_tract_lengths,
    ensure_repo_on_path,
    homogeneous_node_params,
    make_tvb_connectivity,
    quiet_optional_imports,
    quiet_tvb,
    save_state_comparison_plot,
)

ensure_repo_on_path(__file__)

# Importing the package root may probe optional dependencies; keep this example's
# output focused on the simulator run.
with quiet_optional_imports():
    from vbi.simulator import Simulator
    from vbi.simulator.models.mpr import mpr
    from vbi.simulator.spec.coupling import CouplingSpec
    from vbi.simulator.spec.integrator import IntegratorSpec
    from vbi.simulator.spec.monitor import MonitorSpec
    from vbi.simulator.spec.simulation import SimulationSpec


NOTEBOOK_PARAMS = {
    "nn": 6,
    "G": 0.33,
    "dt": 0.01,
    "tau": 1.0,
    "eta": -4.6,
    "Delta": 0.7,
    "J": 14.5,
    "I": 2.0,
    "Gamma": 0.0,
    "cr": 1.0,
    "cv": 0.0,
    "rv_decimate": 10,
    "tract_length": 4.0,
    "speed": 4.0,
}


def build_vbi_spec(method: str) -> SimulationSpec:
    weights = complete_graph_weights(NOTEBOOK_PARAMS["nn"])
    tract_lengths = constant_tract_lengths(weights, NOTEBOOK_PARAMS["tract_length"])
    node_param_names = ("tau", "I", "Delta", "J", "eta", "Gamma", "cr", "cv", "G")
    node_params = homogeneous_node_params(
        n_nodes=weights.shape[0],
        params={name: NOTEBOOK_PARAMS[name] for name in node_param_names},
    )

    return SimulationSpec(
        model=mpr,
        integrator=IntegratorSpec(method=method, dt=NOTEBOOK_PARAMS["dt"]),
        coupling=CouplingSpec(kind="linear", a=1.0, b=0.0),
        monitors=(MonitorSpec(kind="raw"),),
        weights=weights,
        tract_lengths=tract_lengths,
        speed=NOTEBOOK_PARAMS["speed"],
        node_params=node_params,
    )


def run_vbi(duration: float, method: str) -> tuple[np.ndarray, np.ndarray]:
    return Simulator(build_vbi_spec(method), backend="numpy").run(duration)["raw"]


def run_tvb(duration: float, method: str) -> np.ndarray:
    try:
        from tvb.simulator.coupling import Linear
        from tvb.simulator.integrators import EulerDeterministic, HeunDeterministic
        from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
        from tvb.simulator.monitors import Raw
        from tvb.simulator.simulator import Simulator as TVBSimulator
    except ImportError as exc:
        raise RuntimeError("TVB comparison requires the 'tvb' package") from exc

    weights = complete_graph_weights(NOTEBOOK_PARAMS["nn"])
    tract_lengths = constant_tract_lengths(weights, NOTEBOOK_PARAMS["tract_length"])
    n_nodes = weights.shape[0]
    conn = make_tvb_connectivity(
        weights,
        tract_lengths=tract_lengths,
        speed=NOTEBOOK_PARAMS["speed"],
    )

    tvb_model = MontbrioPazoRoxin(
        tau=np.array([NOTEBOOK_PARAMS["tau"]]),
        I=np.array([NOTEBOOK_PARAMS["I"]]),
        Delta=np.array([NOTEBOOK_PARAMS["Delta"]]),
        J=np.array([NOTEBOOK_PARAMS["J"]]),
        eta=np.array([NOTEBOOK_PARAMS["eta"]]),
        Gamma=np.array([NOTEBOOK_PARAMS["Gamma"]]),
        cr=np.array([NOTEBOOK_PARAMS["cr"]]),
        cv=np.array([NOTEBOOK_PARAMS["cv"]]),
    )
    integrator_cls = {
        "euler": EulerDeterministic,
        "heun": HeunDeterministic,
    }[method]

    with quiet_tvb():
        sim = TVBSimulator(
            connectivity=conn,
            model=tvb_model,
            coupling=Linear(a=np.array([NOTEBOOK_PARAMS["G"]])),
            integrator=integrator_cls(dt=NOTEBOOK_PARAMS["dt"]),
            monitors=[Raw()],
            simulation_length=duration,
        ).configure()

        initial_state = np.zeros((tvb_model.nvar, n_nodes, 1))
        initial_state[0, :, 0] = 0.0
        initial_state[1, :, 0] = -2.0
        sim.current_state[:] = initial_state
        sim.history.buffer[:] = initial_state[tvb_model.cvar][np.newaxis, ...]

        (_times, data), = sim.run()

    return data[:, :, :, 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=40.0,
        help="comparison duration in milliseconds",
    )
    parser.add_argument(
        "--method",
        choices=("euler", "heun"),
        default="heun",
        help="deterministic integrator used by both simulators",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=NOTEBOOK_PARAMS["rv_decimate"],
        help="plot every Nth raw sample",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("mpr_tvb_vbi_timeseries.png"),
        help="path for the comparison figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times, vbi_data = run_vbi(duration=args.duration, method=args.method)
    tvb_data = run_tvb(duration=args.duration, method=args.method)

    metrics = comparison_metrics(reference=tvb_data, candidate=vbi_data)

    save_state_comparison_plot(
        times=times,
        left_data=vbi_data,
        right_data=tvb_data,
        out_path=args.output,
        variable_names=("r", "V"),
        title="MPR trajectories: VBI simulator vs TVB",
        decimate=max(1, args.decimate),
    )

    print("MPR TVB comparison using notebook working-point parameters")
    print(f"nodes: {NOTEBOOK_PARAMS['nn']}, G: {NOTEBOOK_PARAMS['G']}, dt: {NOTEBOOK_PARAMS['dt']} ms")
    print(
        "delay: "
        f"{NOTEBOOK_PARAMS['tract_length'] / NOTEBOOK_PARAMS['speed']:.3f} ms "
        f"(tract_length={NOTEBOOK_PARAMS['tract_length']}, speed={NOTEBOOK_PARAMS['speed']})"
    )
    print(f"trajectory shape: {vbi_data.shape}  # (time, variable, node)")
    print(f"max absolute error: {metrics['max_abs']:.6e}")
    print(f"RMS error: {metrics['rms']:.6e}")
    print(f"saved figure: {args.output}")


if __name__ == "__main__":
    main()
