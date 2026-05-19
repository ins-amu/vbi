from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Stuart-Landau (supercritical Hopf) normal form.
# cvar=("x","y") → c_x = coupling[0] for x, c_y = coupling[1] for y.
sup_hopf = ModelSpec(
    name="SupHopf",
    state_variables=(
        StateVar("x", default_init=0.0, noise=True, lower_bound=-5.0, upper_bound=5.0),
        StateVar("y", default_init=0.0, noise=True, lower_bound=-5.0, upper_bound=5.0),
    ),
    parameters=(
        Parameter("a",     -0.5, "local bifurcation parameter (a < 0 → fixed point, a > 0 → limit cycle)"),
        Parameter("omega",  1.0, "angular frequency (rad/ms)"),
    ),
    cvar=("x", "y"),
    dfun_str={
        "x": "(a - x**2 - y**2) * x - omega * y + c_x",
        "y": "(a - x**2 - y**2) * y + omega * x + c_y",
    },
    noise_variables=("x", "y"),
    reference=(
        "Deco G et al. The dynamics of resting fluctuations in the brain: "
        "metastability and its dynamical cortical core. Sci Rep 7:3095, 2017."
    ),
)
