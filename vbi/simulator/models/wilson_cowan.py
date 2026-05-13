from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Defaults match TVB WilsonCowan exactly.
# cvar=[E, I]: TVB cvar=[0, 1]; long-range coupling enters x_e only via c (=c_E=coupling[0]).
# Shifted sigmoid (TVB default shift_sigmoid=True): S(x) - S(0) so that S(0)=0.
# S_e(x_e) = c_e * (1/(1+exp(-a_e*(x_e-b_e))) - 1/(1+exp(a_e*b_e)))
wilson_cowan = ModelSpec(
    name="WilsonCowan",
    state_variables=(
        StateVar("E", default_init=0.0, noise=True, lower_bound=0.0, upper_bound=1.0),
        StateVar("I", default_init=0.0, noise=True, lower_bound=0.0, upper_bound=1.0),
    ),
    parameters=(
        Parameter("c_ee",    12.0, "excitatory to excitatory coupling coefficient"),
        Parameter("c_ei",     4.0, "inhibitory to excitatory coupling coefficient"),
        Parameter("c_ie",    13.0, "excitatory to inhibitory coupling coefficient"),
        Parameter("c_ii",    11.0, "inhibitory to inhibitory coupling coefficient"),
        Parameter("tau_e",   10.0, "excitatory membrane time constant [ms]"),
        Parameter("tau_i",   10.0, "inhibitory membrane time constant [ms]"),
        Parameter("a_e",      1.2, "slope of excitatory sigmoid"),
        Parameter("b_e",      2.8, "position of maximum slope of excitatory sigmoid"),
        Parameter("c_e",      1.0, "amplitude of excitatory sigmoid"),
        Parameter("theta_e",  0.0, "excitatory threshold"),
        Parameter("a_i",      1.0, "slope of inhibitory sigmoid"),
        Parameter("b_i",      4.0, "position of maximum slope of inhibitory sigmoid"),
        Parameter("c_i",      1.0, "amplitude of inhibitory sigmoid"),
        Parameter("theta_i",  0.0, "inhibitory threshold"),
        Parameter("r_e",      1.0, "excitatory refractory period"),
        Parameter("r_i",      1.0, "inhibitory refractory period"),
        Parameter("k_e",      1.0, "maximum value of excitatory response function"),
        Parameter("k_i",      1.0, "maximum value of inhibitory response function"),
        Parameter("P",        0.0, "external stimulus to excitatory population"),
        Parameter("Q",        0.0, "external stimulus to inhibitory population"),
        Parameter("alpha_e",  1.0, "excitatory gain factor"),
        Parameter("alpha_i",  1.0, "inhibitory gain factor"),
    ),
    cvar=("E", "I"),
    dfun_str={
        # x_e = alpha_e*(c_ee*E - c_ei*I + P - theta_e + c)   [c = long-range coupling on E]
        # s_e = c_e*(sigmoid(x_e) - sigmoid(0))                [shifted so S(0)=0]
        "E": (
            "(-E + (k_e - r_e*E) * (c_e * ("
            "1/(1+exp(-a_e*(alpha_e*(c_ee*E - c_ei*I + P - theta_e + c) - b_e)))"
            " - 1/(1+exp(a_e*b_e))))) / tau_e"
        ),
        # x_i = alpha_i*(c_ie*E - c_ii*I + Q - theta_i)        [no long-range coupling]
        # s_i = c_i*(sigmoid(x_i) - sigmoid(0))
        "I": (
            "(-I + (k_i - r_i*I) * (c_i * ("
            "1/(1+exp(-a_i*(alpha_i*(c_ie*E - c_ii*I + Q - theta_i) - b_i)))"
            " - 1/(1+exp(a_i*b_i))))) / tau_i"
        ),
    },
    noise_variables=("E", "I"),
    reference="Wilson HR, Cowan JD. Biophys J. 1972;12(1):1-24. doi:10.1016/S0006-3495(72)86068-5",
)
