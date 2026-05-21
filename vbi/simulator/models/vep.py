import numpy as np
from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Seizure-permittivity 2-D VEP model (simplified Epileptor).
#
# Coupling: Laplacian — G * sum_j W_ij*(x_j - x_i).
# In the framework the coupling layer computes c = G*a*W@x (linear, a=1),
# so the Laplacian term expands to  c - G*row_sum*x.
#
# Required node_params:
#   row_sum = weights.sum(axis=1)   — per-node sum of incoming weights
#
# Per-node parameters (pass via node_params):
#   eta   — node excitability threshold
#   iext  — external forcing per node

vep = ModelSpec(
    name="VEP",
    state_variables=(
        StateVar("x", default_init=-2.0, noise=True),
        StateVar("y", default_init= 3.0, noise=True),
    ),
    parameters=(
        Parameter("tau",      10.0, "time constant of the slow variable y [ms]"),
        Parameter("eta",      -1.5, "excitability threshold — heterogeneous per node via node_params"),
        Parameter("iext",      0.0, "external forcing — heterogeneous per node via node_params"),
        Parameter("G",         1.0, "global coupling strength"),
        Parameter("row_sum",   1.0, "per-node sum of incoming weights; set node_params['row_sum']=weights.sum(axis=1)"),
    ),
    cvar=("x",),
    dfun_str={
        "x": "1.0 - x**3 - 2.0*x**2 - y + iext",
        # Laplacian: G*(W@x - row_sum*x) = c - G*row_sum*x  (c = G*W@x from coupling layer)
        "y": "(4.0*(x - eta) - y - c + G*row_sum*x) / tau",
    },
    noise_variables=("x", "y"),
    reference=(
        "Jirsa VK et al. On the nature of seizure dynamics. Brain 137(8):2210-2230, 2014. "
        "doi:10.1093/brain/awu133"
    ),
    dfun_latex={
        "x": r"1 - x^3 - 2x^2 - y + I_{\rm ext}",
        "y": r"\frac{1}{\tau}\!\left(4(x - \eta) - y - c + G\,r_{\rm sum}\,x\right)",
    },
)
