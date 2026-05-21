from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

linear = ModelSpec(
    name="Linear",
    state_variables=(
        StateVar("x", default_init=0.0, noise=True, lower_bound=-1.0, upper_bound=1.0),
    ),
    parameters=(
        Parameter("gamma", -10.0, "damping coefficient (must be < 0 for stability)"),
    ),
    cvar=("x",),
    dfun_str={
        "x": "gamma * x + c",
    },
    noise_variables=("x",),
    reference="Linear relaxation model — baseline for stability analysis.",
    dfun_latex={
        "x": r"\gamma\,x + c^{\rm net}",
    },
    latex_notes=r"Stable when $\gamma < 0$. Coupling $c^{\rm net} = G\sum_j W_{ij} x_j$.",
)
