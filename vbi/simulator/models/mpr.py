from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Defaults match TVB MontbrioPazoRoxin exactly.
# cvar=[r,V]: both are coupling variables (TVB: Coupling_Term_r, Coupling_Term_V).
# cr=1, cv=0 by default → only r-coupling active unless cv is changed.
# Coupling enters V only: cr*c_r + cv*c_V  (r equation has no coupling term).
mpr = ModelSpec(
    name="MontbrioPopulationRate",
    state_variables=(
        StateVar("r", default_init=0.0, noise=True, lower_bound=0.0),
        StateVar("V", default_init=-2.0, noise=True),
    ),
    parameters=(
        Parameter("tau",    1.0,   "characteristic time constant"),
        Parameter("I",      0.0,   "external current"),
        Parameter("Delta",  0.7,   "half-width of Lorentzian heterogeneity"),
        Parameter("J",      14.5,  "mean synaptic weight"),
        Parameter("eta",   -4.6,   "mean excitability"),
        Parameter("Gamma",  0.0,   "half-width of synaptic weight distribution"),
        Parameter("cr",     1.0,   "coupling weight on r (firing rate)"),
        Parameter("cv",     0.0,   "coupling weight on V (membrane potential)"),
    ),
    cvar=("r", "V"),   # TVB cvar=[0,1]; both variables enter coupling
    dfun_str={
        # c_r = a * sum_j(w_ij * r_j(t-delay))   [injected as c_r by build_dfun]
        # c_V = a * sum_j(w_ij * V_j(t-delay))   [injected as c_V by build_dfun]
        # Global coupling strength goes in CouplingSpec.a (same convention as TVB).
        "r": "(Delta / (pi * tau) + 2 * V * r) / tau",
        "V": "(V**2 - (pi * tau * r)**2 + eta + J * tau * r + I + cr * c_r + cv * c_V) / tau",
    },
    noise_variables=("r", "V"),
    reference="Montbrio, Pazo, Roxin 2015 PLoS Comput Biol 11(12):e1004817",
)
