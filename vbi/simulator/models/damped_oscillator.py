from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Nonlinear damped oscillator (Lotka-Volterra-type).
# Single-node model; no network coupling (set weights=zeros or use N=1).
#
# dx/dt = x - x*y - a*x²
# dy/dt = x*y - y - b*y²

damped_oscillator = ModelSpec(
    name="DampedOscillator",
    state_variables=(
        StateVar("x", default_init=0.5, lower_bound=0.0),
        StateVar("y", default_init=1.0, lower_bound=0.0),
    ),
    parameters=(
        Parameter("a", 0.1,  "x-damping coefficient (ax² term)"),
        Parameter("b", 0.05, "y-damping coefficient (by² term)"),
    ),
    cvar=("x",),
    dfun_str={
        "x": "x - x*y - a*x**2",
        "y": "x*y - y - b*y**2",
    },
    noise_variables=(),
    reference=(
        "Lotka AJ. Elements of Physical Biology. Williams & Wilkins, 1925. "
        "Volterra V. Fluctuations in the Abundance of a Species. Nature 118:558, 1926."
    ),
)
