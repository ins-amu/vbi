from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Jansen-Rit neural-mass model (Jansen & Rit 1995).
# Sigmoid: S(v) = 2*nu_max / (1 + exp(r*(v0 - v)))
# State ordering: y0=pyramidal soma, y1=exc dendritic, y2=inh dendritic,
#                 y3=d(y0)/dt,       y4=d(y1)/dt,      y5=d(y2)/dt
#
# Long-range coupling: use CouplingSpec(kind="jr_sigmoidal") so that
#   c_y1 = W @ S(y1 - y2)  (weighted firing-rate drive; sigmoid applied per
#                            source node before structural weighting).
# The model dfun then applies G * c_y1, giving the TVB/JR convention:
#   G * sum_j W_ij * S(y1_j(t - tau_ji) - y2_j(t - tau_ji))
jansen_rit = ModelSpec(
    name="JansenRit",
    state_variables=(
        StateVar("y0", default_init=0.0),
        StateVar("y1", default_init=0.0),
        StateVar("y2", default_init=0.0),
        StateVar("y3", default_init=0.0),
        StateVar("y4", default_init=0.0, noise=True),
        StateVar("y5", default_init=0.0),
    ),
    parameters=(
        Parameter("A",      3.25,   "maximum amplitude of EPSP [mV]"),
        Parameter("B",      22.0,   "maximum amplitude of IPSP [mV]"),
        Parameter("a",      0.1,    "excitatory time constant [ms^-1]"),
        Parameter("b",      0.05,   "inhibitory time constant [ms^-1]"),
        Parameter("v0",     5.52,   "firing threshold [mV]"),
        Parameter("nu_max", 0.0025, "maximum firing rate [ms^-1]"),
        Parameter("r",      0.56,   "sigmoid steepness [mV^-1]"),
        Parameter("J",      135.0,  "average number of synapses between populations"),
        Parameter("a_1",    1.0,    "synaptic contact probability (exc feedback)"),
        Parameter("a_2",    0.8,    "synaptic contact probability (slow exc feedback)"),
        Parameter("a_3",    0.25,   "synaptic contact probability (inh feedback)"),
        Parameter("a_4",    0.25,   "synaptic contact probability (slow inh feedback)"),
        Parameter("mu",     0.22,   "mean input firing rate"),
        Parameter("G",      1.0,    "global coupling gain"),
    ),
    cvar=("y1", "y2"),
    dfun_str={
        "y0": "y3",
        "y1": "y4",
        "y2": "y5",
        "y3": "A * a * (2*nu_max / (1 + exp(r*(v0 - (y1 - y2))))) - 2*a*y3 - a**2*y0",
        "y4": "A * a * (mu + a_2*J*(2*nu_max/(1+exp(r*(v0 - a_1*J*y0)))) + G*c_y1) - 2*a*y4 - a**2*y1",
        "y5": "B * b * (a_4*J*(2*nu_max/(1+exp(r*(v0 - a_3*J*y0))))) - 2*b*y5 - b**2*y2",
    },
    noise_variables=("y4",),
    reference="Jansen BH, Rit VG. Biol Cybern. 1995;73(4):357-366. doi:10.1007/BF00199471",
    dfun_latex={
        "y0": r"y_3",
        "y1": r"y_4",
        "y2": r"y_5",
        "y3": r"Aa\,S(y_1 - y_2) - 2a\,y_3 - a^2 y_0",
        "y4": r"Aa\!\left(\mu + \alpha_2 J\,S(\alpha_1 J y_0) + G\,c_{y_1}\right) - 2a\,y_4 - a^2 y_1",
        "y5": r"Bb\,\alpha_4 J\,S(\alpha_3 J y_0) - 2b\,y_5 - b^2 y_2",
    },
    latex_notes=(
        r"Sigmoid: $S(v) = \dfrac{2\nu_{\max}}{1 + e^{r(v_0 - v)}}$. "
        r"State variables: $y_0$ pyramidal soma, $y_1$ exc dendritic, $y_2$ inh dendritic; "
        r"$y_3{=}\dot{y}_0,\; y_4{=}\dot{y}_1$ (noisy), $y_5{=}\dot{y}_2$. "
        r"Long-range: $c_{y_1} = \sum_j W_{ij}\,S(y_{1,j}-y_{2,j})$ via "
        r"\texttt{CouplingSpec(kind='jr\_sigmoidal')}; "
        r"the sigmoid is applied per source node before structural weighting, "
        r"then gain $G$ is applied in $\dot{y}_4$."
    ),
)
