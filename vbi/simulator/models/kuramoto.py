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
)
