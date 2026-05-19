from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Ott-Antonsen reduction of QIF theta neurons (2D).
# Intermediate g = k*pi*r is inlined into both dfun expressions.
# cvar=("r","V") → c_r = coupling[0] is the only coupling term used.
coombes_byrne_2d = ModelSpec(
    name="CoombesByrne2D",
    state_variables=(
        StateVar("r", default_init=0.0, noise=True, lower_bound=0.0),
        StateVar("V", default_init=0.0, noise=True, lower_bound=-2.0, upper_bound=1.5),
    ),
    parameters=(
        Parameter("Delta", 1.0,  "Half-width of heterogeneous noise (Lorentzian)"),
        Parameter("v_syn",-4.0,  "Synaptic reversal potential"),
        Parameter("k",     1.0,  "Local coupling / conductance strength"),
        Parameter("eta",   2.0,  "Mean of heterogeneous noise distribution"),
    ),
    cvar=("r", "V"),
    dfun_str={
        # g = k * pi * r  (inlined)
        "r": "Delta / pi + 2*V*r - k*pi*r**2",
        "V": "V**2 - pi**2*r**2 + eta + (v_syn - V)*k*pi*r + c_r",
    },
    noise_variables=("r", "V"),
    reference=(
        "Coombes S, Byrne A. Next generation neural mass models. "
        "In: Nonlinear Dynamics in Computational Neuroscience, Springer, 2019."
    ),
)
