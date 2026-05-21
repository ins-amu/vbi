from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

kuramoto = ModelSpec(
    name="Kuramoto",
    state_variables=(
        StateVar("theta", default_init=0.0, noise=True),
    ),
    parameters=(
        Parameter("omega", 1.0, "natural angular frequency (rad/ms)"),
    ),
    cvar=("theta",),
    dfun_str={
        "theta": "omega + c",
    },
    noise_variables=("theta",),
    reference=(
        "Kuramoto Y. in: H. Arakai (Ed.), International Symposium on Mathematical "
        "Problems in Theoretical Physics. Lecture Notes in Physics 39:420, 1975."
    ),
    dfun_latex={
        "theta": (
            r"\omega_i + \frac{G}{N}\sum_j W_{ij}\,\sin(\theta_j - \theta_i)"
        ),
    },
    latex_notes=(
        r"In code `c` represents the pre-computed coupling term. "
        r"For correct sinusoidal coupling use `CouplingSpec(kind='kuramoto')`, "
        r"which computes $c_i = \frac{G}{N}\sum_j W_{ij}\sin(\theta_j - \theta_i)$. "
        r"With `kind='linear'`, $c_i = G\sum_j W_{ij}\theta_j$ (linear approximation)."
    ),
)
