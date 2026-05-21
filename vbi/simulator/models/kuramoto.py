from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

kuramoto = ModelSpec(
    name="Kuramoto",
    state_variables=(
        StateVar("theta", default_init=0.0, noise=True),
    ),
    parameters=(
        Parameter("omega", 1.0, "natural angular frequency (rad/ms)"),
        Parameter("G",     1.0, "global coupling strength"),
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
            r"\omega_i + \underbrace{\frac{G}{N}\sum_j W_{ij}\,\sin(\theta_j - \theta_i)}_{c_i}"
        ),
    },
    latex_notes=(
        r"$c_i$ is computed by `CouplingSpec(kind='kuramoto')`: "
        r"$c_i = \dfrac{G}{N}\sum_j W_{ij}\sin(\theta_j - \theta_i)$, "
        r"where $W_{ij}$ is the weight **from** source $j$ **to** target $i$ "
        r"(convention: `weights[tgt, src]`), $N$ is the number of nodes, "
        r"and $G$ is the global coupling strength. "
        r"Delayed coupling ($\tau_{ij} > 0$) substitutes $\theta_j(t-\tau_{ij})$ "
        r"for $\theta_j(t)$."
    ),
)
