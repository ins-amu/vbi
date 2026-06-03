"""Compare VBI simulator and TVB Jansen-Rit trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np

from helpers import (
    comparison_metrics,
    complete_graph_weights,
    ensure_repo_on_path,
    homogeneous_node_params,
    make_tvb_connectivity,
    quiet_optional_imports,
    quiet_tvb,
    save_state_comparison_plot,
)

ensure_repo_on_path(__file__)

with quiet_optional_imports():
    from vbi.simulator import Simulator
    from vbi.simulator.models.jansen_rit import jansen_rit
    from vbi.simulator.spec.coupling import CouplingSpec
    from vbi.simulator.spec.integrator import IntegratorSpec
    from vbi.simulator.spec.monitor import MonitorSpec
    from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.connectivity import Connectivity


JR_PARAMS = {
    "nn": 6,
    "coupling_strength": 0.01,
    "dt": 0.05,
    "A": 3.25,
    "B": 22.0,
    "a": 0.1,
    "b": 0.05,
    "v0": 5.52,
    "nu_max": 0.0025,
    "r": 0.56,
    "J": 135.0,
    "a_1": 1.0,
    "a_2": 1.0,
    "a_3": 0.25,
    "a_4": 0.25,
    "mu": 0.04, # 0.24
    "decimate": 20,
}


def demo_weights(n_nodes: int) -> np.ndarray:
    weights = complete_graph_weights(n_nodes)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return weights / row_sums


def build_vbi_spec(method: str) -> SimulationSpec:
    weights = demo_weights(JR_PARAMS["nn"])
    param_names = (
        "A", "B", "a", "b", "v0", "nu_max", "r", "J",
        "a_1", "a_2", "a_3", "a_4", "mu",
    )
    node_params = homogeneous_node_params(
        n_nodes=weights.shape[0],
        params={name: JR_PARAMS[name] for name in param_names},
        scalar_names=(),
    )

    # This demo compares deterministic VBI and TVB trajectories.
    #
    # For a VBI-only stochastic run, the JR model already marks y4 as the
    # noisy state variable, so switch only the integrator:
    #
    #   IntegratorSpec(
    #       method=method,
    #       dt=JR_PARAMS["dt"],
    #       stochastic=True,
    #       noise_nsig=np.array([0.001]),  # σ, one value per noise_variable
    #       noise_style="amplitude",       # σ * sqrt(dt) * N(0,1)
    #       noise_seed=42,
    #   )
    #
    # For a stochastic TVB comparison, run_tvb() must also use TVB's
    # stochastic integrator and Additive noise. Use noise_style="tvb" on
    # the VBI side if you want TVB-compatible nsig semantics.
    return SimulationSpec(
        model=jansen_rit,
        integrator=IntegratorSpec(method=method, dt=JR_PARAMS["dt"]),
        coupling=CouplingSpec(kind="linear", a=JR_PARAMS["coupling_strength"]),
        monitors=(MonitorSpec(kind="raw"),),
        connectivity=Connectivity(weights, speed=1.0),
        node_params=node_params,
    )


def run_vbi(duration: float, method: str) -> tuple[np.ndarray, np.ndarray]:
    return Simulator(build_vbi_spec(method), backend="numpy").run(duration)["raw"]


def run_tvb(duration: float, method: str) -> np.ndarray:
    try:
        from tvb.simulator.coupling import Linear
        from tvb.simulator.integrators import EulerDeterministic, HeunDeterministic
        from tvb.simulator.models.jansen_rit import JansenRit
        from tvb.simulator.monitors import Raw
        from tvb.simulator.simulator import Simulator as TVBSimulator
    except ImportError as exc:
        raise RuntimeError("TVB comparison requires the 'tvb' package") from exc

    weights = demo_weights(JR_PARAMS["nn"])
    n_nodes = weights.shape[0]
    conn = make_tvb_connectivity(weights, speed=1.0)

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

    with quiet_tvb():
        sim = TVBSimulator(
            connectivity=conn,
            model=tvb_model,
            coupling=Linear(a=np.array([JR_PARAMS["coupling_strength"]])),
            integrator=integrator_cls(dt=JR_PARAMS["dt"]),
            monitors=[Raw()],
            simulation_length=duration,
        ).configure()

        initial_state = np.zeros((tvb_model.nvar, n_nodes, 1))
        sim.current_state[:] = initial_state
        sim.history.buffer[:] = initial_state[tvb_model.cvar][np.newaxis, ...]

        (_times, data), = sim.run()

    return data[:, :, :, 0]


def comparison_view(data: np.ndarray, n_plot_nodes: int) -> np.ndarray:
    eeg = data[:, 1, :] - data[:, 2, :]
    return np.stack((data[:, 0, :n_plot_nodes], eeg[:, :n_plot_nodes]), axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=25.0,
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
        default=JR_PARAMS["decimate"],
        help="plot every Nth raw sample",
    )
    parser.add_argument(
        "--plot-nodes",
        type=int,
        default=6,
        help="number of nodes to include in the overlay plot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("outputs") / "jansen_rit_tvb_vbi_timeseries.png",
        help="path for the comparison figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times, vbi_data = run_vbi(duration=args.duration, method=args.method)
    tvb_data = run_tvb(duration=args.duration, method=args.method)
    metrics = comparison_metrics(reference=tvb_data, candidate=vbi_data)
    scale = max(float(np.nanmax(np.abs(tvb_data))), 1.0)
    relative_max_error = metrics["max_abs"] / scale

    save_state_comparison_plot(
        times=times,
        left_data=comparison_view(vbi_data, args.plot_nodes),
        right_data=comparison_view(tvb_data, args.plot_nodes),
        out_path=args.output,
        variable_names=("y0", "y1 - y2"),
        title="Jansen-Rit trajectories: VBI simulator vs TVB",
        decimate=args.decimate,
    )

    print("Jansen-Rit TVB comparison")
    print(
        f"nodes: {JR_PARAMS['nn']}, coupling: {JR_PARAMS['coupling_strength']}, "
        f"dt: {JR_PARAMS['dt']} ms, mu: {JR_PARAMS['mu']}, C1: {JR_PARAMS['a_2'] * JR_PARAMS['J']}"
    )
    print(f"trajectory shape: {vbi_data.shape}  # (time, variable, node)")
    print(f"max absolute error: {metrics['max_abs']:.6e}")
    print(f"RMS error: {metrics['rms']:.6e}")
    print(f"relative max error: {relative_max_error:.6e}")
    print(f"saved figure: {args.output}")


if __name__ == "__main__":
    main()
