from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Ott-Antonsen QIF reduction with Synaptic Depression adaptation.
# cvar=("r","V") - only r and V enter the long-range coupling (cr, cv weights).
# c_r = coupling[0], c_V = coupling[1].
gast_sd = ModelSpec(
    name="GastSchmidtKnosche_SD",
    state_variables=(
        StateVar("r", default_init=0.0, noise=True,  lower_bound=0.0),
        StateVar("V", default_init=0.0, noise=True,  lower_bound=-3.0, upper_bound=0.3),
        StateVar("A", default_init=0.0, noise=False, lower_bound=0.0,  upper_bound=0.4),
        StateVar("B", default_init=0.0, noise=False, lower_bound=-0.2, upper_bound=0.3),
    ),
    parameters=(
        Parameter("tau",   1.0,      "Characteristic time constant"),
        Parameter("tau_A", 10.0,     "Adaptation time scale"),
        Parameter("alpha", 0.5,      "Adaptation rate"),
        Parameter("I",     0.0,      "External homogeneous current"),
        Parameter("Delta", 2.0,      "Half-width of heterogeneous noise"),
        Parameter("J",     21.2132,  "Synaptic weight"),
        Parameter("eta",  -6.0,      "Mean of heterogeneous noise distribution"),
        Parameter("cr",    1.0,      "Weight on r-channel long-range coupling"),
        Parameter("cv",    0.0,      "Weight on V-channel long-range coupling"),
    ),
    cvar=("r", "V"),
    dfun_str={
        "r": "1/tau * (Delta/(pi*tau) + 2*V*r)",
        "V": "1/tau * (V**2 - pi**2*tau**2*r**2 + eta + J*tau*r*(1-A) + I + cr*c_r + cv*c_V)",
        "A": "1/tau_A * B",
        "B": "1/tau_A * (-2*B - A + alpha*r)",
    },
    noise_variables=("r", "V"),
    reference=(
        "Gast R, Schmidt H, Knosche TR. A mean-field description of bursting dynamics "
        "in spiking neural networks with short-term adaptation. "
        "Neural Computation 32(9):1615-1634, 2020."
    ),
    dfun_latex={
        "r": r"\frac{1}{\tau}\!\left(\frac{\Delta}{\pi\tau} + 2Vr\right)",
        "V": r"\frac{1}{\tau}\!\left(V^2 - (\pi\tau r)^2 + \eta + J\tau r(1-A) + I"
             r" + c_r\,c_r^{\rm net} + c_v\,c_V^{\rm net}\right)",
        "A": r"\frac{B}{\tau_A}",
        "B": r"\frac{1}{\tau_A}\!\left(-2B - A + \alpha r\right)",
    },
)
