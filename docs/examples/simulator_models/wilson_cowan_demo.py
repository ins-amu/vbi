"""Reproduce the first Wilson-Cowan SDE notebook visualization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np

from helpers import (
    ensure_repo_on_path,
    homogeneous_node_params,
    quiet_optional_imports,
)

ensure_repo_on_path(__file__)

with quiet_optional_imports():
    from vbi.simulator import Simulator
    from vbi.simulator.models.wilson_cowan import wilson_cowan
    from vbi.simulator.spec.coupling import CouplingSpec
    from vbi.simulator.spec.integrator import IntegratorSpec
    from vbi.simulator.spec.monitor import MonitorSpec
    from vbi.simulator.spec.simulation import SimulationSpec
from vbi.simulator.spec.connectivity import Connectivity


WC_PARAMS = {
    "nn": 2,
    "coupling_strength": 0.0,
    "dt": 0.05,
    "t_cut": 101.0,
    "noise_amp": 0.0005,
    "noise_seed": 42,
    "c_ee": 16.0,
    "c_ei": 12.0,
    "c_ie": 15.0,
    "c_ii": 3.0,
    "tau_e": 8.0,
    "tau_i": 8.0,
    "a_e": 1.3,
    "b_e": 4.0,
    "c_e": 1.0,
    "theta_e": 0.0,
    "a_i": 2.0,
    "b_i": 3.7,
    "c_i": 1.0,
    "theta_i": 0.0,
    "r_e": 1.0,
    "r_i": 1.0,
    "k_e": 0.994,
    "k_i": 0.999,
    "P": 1.025,
    "Q": 0.0,
    "alpha_e": 1.0,
    "alpha_i": 1.0,
    "shift_sigmoid": 0.0,
    "decimate": 1,
}


def build_vbi_spec(
    method: str,
    stochastic: bool = True,
    param_overrides: dict[str, float] | None = None,
) -> SimulationSpec:
    weights = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    params = dict(WC_PARAMS)
    if param_overrides:
        params.update(param_overrides)
    param_names = (
        "c_ee", "c_ei", "c_ie", "c_ii", "tau_e", "tau_i",
        "a_e", "b_e", "c_e", "theta_e", "a_i", "b_i", "c_i",
        "theta_i", "r_e", "r_i", "k_e", "k_i", "P", "Q",
        "alpha_e", "alpha_i", "shift_sigmoid",
    )
    node_params = homogeneous_node_params(
        n_nodes=weights.shape[0],
        params={name: params[name] for name in param_names},
        scalar_names=(),
    )

    return SimulationSpec(
        model=wilson_cowan,
        integrator=IntegratorSpec(
            method=method,
            dt=params["dt"],
            stochastic=stochastic,
            noise_nsig=np.array([params["noise_amp"], params["noise_amp"]]),
            noise_style="amplitude",
            noise_seed=params["noise_seed"],
        ),
        coupling=CouplingSpec(kind="linear", a=params["coupling_strength"]),
        monitors=(MonitorSpec(kind="raw"),),
        connectivity=Connectivity(weights, speed=1.0),
        node_params=node_params,
    )


def run_vbi(
    duration: float,
    method: str,
    stochastic: bool = True,
    param_overrides: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    spec = build_vbi_spec(method, stochastic=stochastic, param_overrides=param_overrides)
    return Simulator(spec, backend="numpy").run(duration)["raw"]


def spectrum(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Welch spectrum for a one-dimensional signal."""
    signal = signal - np.mean(signal)
    try:
        from scipy.signal import welch
    except ImportError:
        freqs = np.fft.rfftfreq(signal.size, d=dt / 1000.0)
        power = np.abs(np.fft.rfft(signal)) ** 2 / max(signal.size, 1)
        return freqs, power

    nperseg = min(4096, signal.size)
    return welch(signal, fs=1000.0 / dt, nperseg=nperseg)


def after_cut(times: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop the initial transient used by the original notebook examples."""
    keep = times >= WC_PARAMS["t_cut"]
    if not np.any(keep):
        return times, data
    return times[keep], data[keep]

def save_single_run_plot(
    times: np.ndarray,
    data: np.ndarray,
    out_path: Path,
    decimate: int,
) -> None:
    """Save time-series, spectrum, and phase-plane views for one run."""
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    decimate = max(1, decimate)
    post_times, post_data = after_cut(times, data)
    post_e = post_data[:, 0, 0]
    post_i = post_data[:, 1, 0]
    freqs_e, power_e = spectrum(post_e, WC_PARAMS["dt"])
    freqs_i, power_i = spectrum(post_i, WC_PARAMS["dt"])

    fig = plt.figure(constrained_layout=True, figsize=(11, 6))
    axes = fig.subplot_mosaic("AA\nBC")
    axes["A"].plot(post_times[::decimate], post_e[::decimate], color="tab:red", lw=0.8, label="E")
    axes["A"].plot(post_times[::decimate], post_i[::decimate], color="tab:blue", lw=0.8, label="I")
    axes["A"].set_ylabel("activity")
    axes["A"].set_title("Wilson-Cowan stochastic trajectory")
    axes["A"].legend(frameon=False)
    axes["A"].grid(True, alpha=0.25)

    axes["B"].plot(freqs_e, power_e, color="tab:red", lw=1.0, label="E")
    axes["B"].plot(freqs_i, power_i, color="tab:blue", lw=1.0, label="I")
    axes["B"].set_xlim(0, 100)
    axes["B"].set_xlabel("frequency [Hz]")
    axes["B"].set_ylabel("power")
    axes["B"].legend(frameon=False)
    axes["B"].grid(True, alpha=0.25)

    axes["C"].plot(post_e[::decimate], post_i[::decimate], color="black", lw=0.6, alpha=0.75)
    axes["C"].set_xlabel("E")
    axes["C"].set_ylabel("I")
    axes["C"].grid(True, alpha=0.25)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=2000.0,
        help="simulation duration in milliseconds",
    )
    parser.add_argument(
        "--method",
        choices=("euler", "heun"),
        default="heun",
        help="stochastic integrator drift method",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=WC_PARAMS["decimate"],
        help="plot every Nth raw sample",
    )
    parser.add_argument(
        "--single-output",
        type=Path,
        default=Path(__file__).with_name("outputs") / "wilson_cowan_notebook_single_run.png",
        help="path for the single-run figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times, data = run_vbi(duration=args.duration, method=args.method)

    save_single_run_plot(
        times=times,
        data=data,
        out_path=args.single_output,
        decimate=args.decimate,
    )

    _post_times, post_data = after_cut(times, data)
    freqs, power = spectrum(post_data[:, 0, 0], WC_PARAMS["dt"])
    nonzero = freqs > 0.0
    peak_freq = float(freqs[nonzero][np.argmax(power[nonzero])]) if np.any(nonzero) else 0.0

    print("Wilson-Cowan notebook-style stochastic simulator demo")
    print(
        f"nodes: {WC_PARAMS['nn']}, coupling: {WC_PARAMS['coupling_strength']}, "
        f"P: {WC_PARAMS['P']}, dt: {WC_PARAMS['dt']} ms, noise_amp: {WC_PARAMS['noise_amp']}"
    )
    print(f"trajectory shape: {data.shape}  # (time, variable, node)")
    print(f"post-cut shape: {post_data.shape}  # t >= {WC_PARAMS['t_cut']} ms")
    print(f"single-run peak E frequency: {peak_freq:.3f} Hz")
    print(f"saved figure: {args.single_output}")


if __name__ == "__main__":
    main()
