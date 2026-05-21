import numpy as np
from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# Stuart-Landau (supercritical Hopf) oscillator — Cartesian form with Laplacian coupling.
# Equivalent to the SL_sde numba model; uses scalar a and omega (same for all nodes).
# For per-node bifurcation/frequency variation use ghb instead.
#
# Complex SDE: dz/dt = z*(a + j*omega - |z|²) + G*sum_j W_ij*(z_j - z_i)
# Expanded in Cartesian coordinates (z = x + jy):
#   dx/dt = (a - x²-y²)*x - omega*y + G*(W@x - row_sum*x)
#         = (a - x²-y²)*x - omega*y + c_x - G*row_sum*x
#   dy/dt = (a - x²-y²)*y + omega*x + G*(W@y - row_sum*y)
#         = (a - x²-y²)*y + omega*x + c_y - G*row_sum*y
#
# Required node_params:
#   row_sum = weights.sum(axis=1)   — per-node sum of incoming weights

sl = ModelSpec(
    name="StuartLandau",
    state_variables=(
        StateVar("x", default_init=0.01, noise=True, lower_bound=-5.0, upper_bound=5.0),
        StateVar("y", default_init=0.01, noise=True, lower_bound=-5.0, upper_bound=5.0),
    ),
    parameters=(
        Parameter("a",       0.1,              "bifurcation parameter (a<0: fixed point, a>0: limit cycle)"),
        Parameter("omega",   2.0*np.pi*0.040,  "angular frequency [rad/ms]; 2π×40 Hz ≈ 0.2513"),
        Parameter("G",       0.0,              "global coupling strength"),
        Parameter("row_sum", 1.0,              "per-node sum of incoming weights; set node_params['row_sum']=weights.sum(axis=1)"),
    ),
    cvar=("x", "y"),
    dfun_str={
        # Laplacian: G*(W@x - row_sum*x) = c_x - G*row_sum*x
        "x": "(a - x**2 - y**2)*x - omega*y + c_x - G*row_sum*x",
        "y": "(a - x**2 - y**2)*y + omega*x + c_y - G*row_sum*y",
    },
    noise_variables=("x", "y"),
    reference=(
        "Stuart A, Landau L. On the Problem of Turbulence. Dokl. Akad. Nauk USSR 44:311, 1944. "
        "Deco G et al. Sci Rep 7:3095, 2017. doi:10.1038/s41598-017-03073-5"
    ),
    dfun_latex={
        "x": r"(a - x^2 - y^2)\,x - \omega y + c_x - G\,r_{\rm sum}\,x",
        "y": r"(a - x^2 - y^2)\,y + \omega x + c_y - G\,r_{\rm sum}\,y",
    },
)
