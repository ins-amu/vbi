from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Defaults match TVB ReducedWongWangExcInh (Deco et al. 2014) exactly.
# cvar=[S_e]: only excitatory gating couples long-range (TVB cvar=[0]).
# G is extracted by the VBI simulator to pre-scale coupling, so c already equals
# G * sum_j(w_ij * S_e_j) — do NOT multiply by G again in dfun_str.
# Transfer function: H(x) = x / (1 - exp(-d*x))  where x = a*input - b
#
# VBI naming → TVB naming:
#   a_e, b_e, d_e, gamma_e, tau_e, w_p, W_e, J_N  (excitatory)
#   a_i, b_i, d_i, gamma_i, tau_i, J_i, W_i        (inhibitory)
#   I_o = I_0 (external input), I_ext (stimulus), lamda (inh LRC scale)
wong_wang_exc_inh = ModelSpec(
    name="ReducedWongWangExcInh",
    state_variables=(
        StateVar("S_e", default_init=0.05, noise=True, lower_bound=0.0, upper_bound=1.0),
        StateVar("S_i", default_init=0.05, noise=True, lower_bound=0.0, upper_bound=1.0),
    ),
    parameters=(
        Parameter("a_e",     310.0,          "excitatory gain [n/C]"),
        Parameter("b_e",     125.0,          "excitatory threshold [Hz]"),
        Parameter("d_e",     0.160,          "excitatory shape parameter [s]"),
        Parameter("gamma_e", 0.641 / 1000,   "excitatory kinetic parameter [ms^-1]"),
        Parameter("tau_e",   100.0,          "excitatory NMDA decay time constant [ms]"),
        Parameter("w_p",     1.4,            "local excitatory recurrence"),
        Parameter("W_e",     1.0,            "excitatory external input weight"),
        Parameter("J_N",     0.15,           "NMDA synaptic current [nA]"),
        Parameter("a_i",     615.0,          "inhibitory gain [n/C]"),
        Parameter("b_i",     177.0,          "inhibitory threshold [Hz]"),
        Parameter("d_i",     0.087,          "inhibitory shape parameter [s]"),
        Parameter("gamma_i", 1.0 / 1000,     "inhibitory kinetic parameter [ms^-1]"),
        Parameter("tau_i",   10.0,           "inhibitory GABA decay time constant [ms]"),
        Parameter("J_i",     1.0,            "local inhibitory current [nA]"),
        Parameter("W_i",     0.7,            "inhibitory external input weight"),
        Parameter("I_o",     0.382,          "effective external input [nA]"),
        Parameter("I_ext",   0.0,            "additional stimulus current [nA]"),
        Parameter("lamda",   0.0,            "inhibitory long-range coupling scale"),
        Parameter("G",       2.0,            "global coupling (applied by simulator to c)"),
    ),
    cvar=("S_e",),
    dfun_str={
        # x_e = a_e*(w_p*J_N*S_e - J_i*S_i + W_e*I_o + J_N*c + I_ext) - b_e
        # H_e = x_e / (1 - exp(-d_e*x_e))
        # c already equals G*sum_j(w_ij*S_e_j) — G is pre-applied by VBI coupling
        "S_e": (
            "-(S_e/tau_e) + (1-S_e) * gamma_e * "
            "((a_e*(w_p*J_N*S_e - J_i*S_i + W_e*I_o + J_N*c + I_ext) - b_e) / "
            "(1 - exp(-d_e*(a_e*(w_p*J_N*S_e - J_i*S_i + W_e*I_o + J_N*c + I_ext) - b_e))))"
        ),
        # x_i = a_i*(J_N*S_e - S_i + W_i*I_o + lamda*J_N*c) - b_i
        # H_i = x_i / (1 - exp(-d_i*x_i))
        "S_i": (
            "-(S_i/tau_i) + gamma_i * "
            "((a_i*(J_N*S_e - S_i + W_i*I_o + lamda*J_N*c) - b_i) / "
            "(1 - exp(-d_i*(a_i*(J_N*S_e - S_i + W_i*I_o + lamda*J_N*c) - b_i))))"
        ),
    },
    noise_variables=("S_e", "S_i"),
    reference="Deco G et al. J Neurosci. 2014;34(23):7886-7898. doi:10.1523/JNEUROSCI.5068-13.2014",
)
