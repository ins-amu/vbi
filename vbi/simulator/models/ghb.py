import numpy as np
from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Generic Hopf Bifurcation (Stuart-Landau) oscillator - Cartesian form.
# Neural part of the GHB_sde numba model; BOLD is handled by BoldMonitor.
#
# Coupling: Laplacian - G * sum_j W_ij*(x_j - x_i) on both x and y.
# With c_x = G*W@x and c_y = G*W@y (linear coupling, a=1):
#   G*gx = c_x - G*row_sum*x
#   G*gy = c_y - G*row_sum*y
#
# Per-node parameters (pass via node_params to vary across nodes):
#   eta     - bifurcation parameter per node
#   omega   - angular frequency per node [rad/ms]
#   row_sum - sum of incoming weights per node; set weights.sum(axis=1)

ghb = ModelSpec(
    name="GHB",
    state_variables=(
        StateVar("x", default_init=0.0, noise=True, lower_bound=-5.0, upper_bound=5.0),
        StateVar("y", default_init=0.0, noise=True, lower_bound=-5.0, upper_bound=5.0),
    ),
    parameters=(
        Parameter("eta",     0.1,                  "bifurcation parameter (eta<0: fixed point, eta>0: limit cycle) - per node via node_params"),
        Parameter("omega",   2.0*np.pi*0.040,      "angular frequency [rad/ms]; 2π×40 Hz ≈ 0.2513 - per node via node_params"),
        Parameter("G",       1.0,                  "global coupling strength"),
        Parameter("row_sum", 1.0,                  "per-node sum of incoming weights; set node_params['row_sum']=weights.sum(axis=1)"),
    ),
    cvar=("x", "y"),
    dfun_str={
        # Laplacian: G*(W@x - row_sum*x) = c_x - G*row_sum*x
        "x": "(eta - x**2 - y**2)*x - omega*y + c_x - G*row_sum*x",
        "y": "(eta - x**2 - y**2)*y + omega*x + c_y - G*row_sum*y",
    },
    noise_variables=("x", "y"),
    reference=(
        "Deco G et al. The dynamics of resting fluctuations in the brain: "
        "metastability and its dynamical cortical core. Sci Rep 7:3095, 2017. "
        "doi:10.1038/s41598-017-03073-5"
    ),
    dfun_latex={
        "x": r"(\eta - x^2 - y^2)\,x - \omega y + c_x - G\,r_{\rm sum}\,x",
        "y": r"(\eta - x^2 - y^2)\,y + \omega x + c_y - G\,r_{\rm sum}\,y",
    },
)
