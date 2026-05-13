from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Defaults match TVB JansenRit exactly.
# cvar=[y1, y2]: TVB cvar=[1, 2]; long-range coupling enters y4 as c (=c_y1=coupling[0]).
# Sigmoid: S(v) = 2*nu_max / (1 + exp(r*(v0 - v)))
# State ordering: y0=pyramidal soma, y1=exc dendritic, y2=inh dendritic,
#                 y3=d(y0)/dt,       y4=d(y1)/dt,      y5=d(y2)/dt
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
    ),
    cvar=("y1", "y2"),
    dfun_str={
        "y0": "y3",
        "y1": "y4",
        "y2": "y5",
        "y3": "A * a * (2*nu_max / (1 + exp(r*(v0 - (y1 - y2))))) - 2*a*y3 - a**2*y0",
        "y4": "A * a * (mu + a_2*J*(2*nu_max/(1+exp(r*(v0 - a_1*J*y0)))) + c) - 2*a*y4 - a**2*y1",
        "y5": "B * b * (a_4*J*(2*nu_max/(1+exp(r*(v0 - a_3*J*y0))))) - 2*b*y5 - b**2*y2",
    },
    noise_variables=("y4",),
    reference="Jansen BH, Rit VG. Biol Cybern. 1995;73(4):357-366. doi:10.1007/BF00199471",
)
