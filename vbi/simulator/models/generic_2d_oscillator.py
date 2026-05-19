from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# TVB parameter 'c' renamed to 'c_coeff' to avoid collision with the
# coupling alias 'c = coupling[0]' injected by dfun codegen.
generic_2d_oscillator = ModelSpec(
    name="Generic2dOscillator",
    state_variables=(
        StateVar("V", default_init=0.0, noise=True, lower_bound=-2.0, upper_bound=4.0),
        StateVar("W", default_init=0.0, noise=True, lower_bound=-6.0, upper_bound=6.0),
    ),
    parameters=(
        Parameter("tau",     1.0,   "time-scale hierarchy (tau > 1 → V faster than W)"),
        Parameter("I",       0.0,   "constant baseline external current"),
        Parameter("a",      -2.0,   "intercept of the cubic V-nullcline"),
        Parameter("b",     -10.0,   "linear slope of the V-nullcline"),
        Parameter("c_coeff", 0.0,   "parabolic coefficient of the V-nullcline (TVB 'c')"),
        Parameter("d",       0.02,  "overall temporal scale factor"),
        Parameter("e",       3.0,   "quadratic term of the cubic V-nullcline"),
        Parameter("f",       1.0,   "cubic term of the cubic V-nullcline"),
        Parameter("g",       0.0,   "linear term of the cubic V-nullcline"),
        Parameter("alpha",   1.0,   "coupling from W to V"),
        Parameter("beta",    1.0,   "self-damping of W"),
        Parameter("gamma",   1.0,   "gain applied to I and afferent coupling"),
    ),
    cvar=("V",),
    dfun_str={
        "V": "d * tau * (alpha*W - f*V**3 + e*V**2 + g*V + gamma*I + gamma*c)",
        "W": "d / tau * (a + b*V + c_coeff*V**2 - beta*W)",
    },
    noise_variables=("V", "W"),
    reference=(
        "FitzHugh R. Impulses and physiological states in theoretical models of "
        "nerve membrane. Biophys J 1(6):445, 1961. "
        "Nagumo J et al. An active pulse transmission line simulating nerve axon. "
        "Proc IRE 50:2061, 1962."
    ),
)
