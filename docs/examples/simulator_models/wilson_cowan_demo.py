"""Compare VBI simulator and TVB Wilson-Cowan trajectories."""

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
    from vbi.simulator.models.wilson_cowan import wilson_cowan
    from vbi.simulator.spec.coupling import CouplingSpec
    from vbi.simulator.spec.integrator import IntegratorSpec
    from vbi.simulator.spec.monitor import MonitorSpec
    from vbi.simulator.spec.simulation import SimulationSpec


WC_PARAMS = {
    "nn": 6,
    "coupling_strength": 0.15,
    "dt": 0.01,
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
    "decimate": 100,
}


def demo_weights(n_nodes: int) -> np.ndarray:
    """Return row-normalized complete-graph weights for a compact demo."""
    weights = complete_graph_weights(n_nodes)
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return weights / row_sums


def build_vbi_spec(method: str) -> SimulationSpec:
    weights = demo_weights(WC_PARAMS["nn"])
    param_names = (
        "c_ee", "c_ei", "c_ie", "c_ii", "tau_e", "tau_i",
        "a_e", "b_e", "c_e", "theta_e", "a_i", "b_i", "c_i",
        "theta_i", "r_e", "r_i", "k_e", "k_i", "P", "Q",
        "alpha_e", "alpha_i",
    )
    node_params = homogeneous_node_params(
        n_nodes=weights.shape[0],
        params={name: WC_PARAMS[name] for name in param_names},
        scalar_names=(),
    )

    return SimulationSpec(
        model=wilson_cowan,
        integrator=IntegratorSpec(method=method, dt=WC_PARAMS["dt"]),
        coupling=CouplingSpec(kind="linear", a=WC_PARAMS["coupling_strength"]),
        monitors=(MonitorSpec(kind="raw"),),
        weights=weights,
        tract_lengths=np.zeros_like(weights),
        speed=1.0,
        node_params=node_params,
    )


def run_vbi(duration: float, method: str) -> tuple[np.ndarray, np.ndarray]:
    return Simulator(build_vbi_spec(method), backend="numpy").run(duration)["raw"]


def run_tvb(duration: float, method: str) -> np.ndarray:
    try:
        from tvb.simulator.coupling import Linear
        from tvb.simulator.integrators import EulerDeterministic, HeunDeterministic
        from tvb.simulator.models.wilson_cowan import WilsonCowan
        from tvb.simulator.monitors import Raw
        from tvb.simulator.simulator import Simulator as TVBSimulator
    except ImportError as exc:
        raise RuntimeError("TVB comparison requires the 'tvb' package") from exc

    weights = demo_weights(WC_PARAMS["nn"])
    n_nodes = weights.shape[0]
    conn = make_tvb_connectivity(weights, speed=1.0)

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

    with quiet_tvb():
        sim = TVBSimulator(
            connectivity=conn,
            model=tvb_model,
            coupling=Linear(a=np.array([WC_PARAMS["coupling_strength"]])),
            integrator=integrator_cls(dt=WC_PARAMS["dt"]),
            monitors=[Raw()],
            simulation_length=duration,
        ).configure()

        initial_state = np.zeros((tvb_model.nvar, n_nodes, 1))
        sim.current_state[:] = initial_state
        sim.history.buffer[:] = initial_state[tvb_model.cvar][np.newaxis, ...]

        (_times, data), = sim.run()

    return data[:, :, :, 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=100.0,
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
        default=WC_PARAMS["decimate"],
        help="plot every Nth raw sample",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("outputs") / "wilson_cowan_tvb_vbi_timeseries.png",
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
        left_data=vbi_data,
        right_data=tvb_data,
        out_path=args.output,
        variable_names=("E", "I"),
        title="Wilson-Cowan trajectories: VBI simulator vs TVB",
        decimate=args.decimate,
    )

    print("Wilson-Cowan TVB comparison")
    print(
        f"nodes: {WC_PARAMS['nn']}, coupling: {WC_PARAMS['coupling_strength']}, "
        f"dt: {WC_PARAMS['dt']} ms"
    )
    print(f"trajectory shape: {vbi_data.shape}  # (time, variable, node)")
    print(f"max absolute error: {metrics['max_abs']:.6e}")
    print(f"RMS error: {metrics['rms']:.6e}")
    print(f"relative max error: {relative_max_error:.6e}")
    print(f"saved figure: {args.output}")


if __name__ == "__main__":
    main()
