from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Defaults match TVB ReducedWongWang (Deco et al. 2013) exactly.
# cvar=[S]: long-range coupling enters via J_N*c in the effective input x.
# S is bounded [0,1] (NMDA gating variable).
# Transfer function: H(x) = (a*x - b) / (1 - exp(-d*(a*x - b)))
reduced_wong_wang = ModelSpec(
    name="ReducedWongWang",
    state_variables=(
        StateVar("S", default_init=0.05, noise=True, lower_bound=0.0, upper_bound=1.0),
    ),
    parameters=(
        Parameter("a",     0.270,  "input gain [n/C]"),
        Parameter("b",     0.108,  "input shift [kHz]"),
        Parameter("d",     154.0,  "transfer function shape [ms]"),
        Parameter("gamma", 0.641,  "kinetic parameter"),
        Parameter("tau_s", 100.0,  "NMDA decay time constant [ms]"),
        Parameter("w",     0.6,    "local excitatory recurrence"),
        Parameter("J_N",   0.2609, "synaptic coupling strength [nA]"),
        Parameter("I_o",   0.33,   "effective external input [nA]"),
    ),
    cvar=("S",),
    dfun_str={
        # x = w*J_N*S + I_o + J_N*c
        # H = (a*x - b) / (1 - exp(-d*(a*x - b)))
        # dS/dt = -(S/tau_s) + (1-S)*H*gamma
        "S": (
            "-(S/tau_s) + (1-S) * gamma * "
            "((a*(w*J_N*S + I_o + J_N*c) - b) / "
            "(1 - exp(-d*(a*(w*J_N*S + I_o + J_N*c) - b))))"
        ),
    },
    noise_variables=("S",),
    reference="Deco G et al. J Neurosci. 2013;32(27):11239-11252. doi:10.1523/JNEUROSCI.1091-12.2013",
    dfun_latex={
        "S": r"-\frac{S}{\tau_s} + (1-S)\,\gamma\,H(x)",
    },
    latex_notes=(
        r"Input: $x = wJ_N S + I_o + J_N c^{\rm net}$. "
        r"Transfer function: $H(x) = \dfrac{ax - b}{1 - e^{-d(ax-b)}}$."
    ),
)
