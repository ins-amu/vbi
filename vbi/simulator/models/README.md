# VBI Simulator Models

All models are defined as **declarative `ModelSpec` objects** - no subclassing required.
A model defines its state variables, parameters, coupling variables, and dfun equations
as plain Python strings. The simulator backends (NumPy, Numba, C++, JAX) compile or
execute these equations automatically.

## Quick reference

| Model object | Name | SVs | Dim | cvar | Noise vars | Bounds | Reference |
|---|---|---|---|---|---|---|---|
| `mpr` | MontbrioPopulationRate | r, V | 2 | r, V | r, V | r ≥ 0 | Montbrio et al. 2015 |
| `jansen_rit` | JansenRit | y0–y5 | 6 | y1, y2 | y4 | - | Jansen & Rit 1995 |
| `wilson_cowan` | WilsonCowan | E, I | 2 | E, I | E, I | E,I ∈ [0,1] | Wilson & Cowan 1972 |
| `reduced_wong_wang` | ReducedWongWang | S | 1 | S | S | S ∈ [0,1] | Deco et al. 2013 |
| `wong_wang_exc_inh` | ReducedWongWangExcInh | S\_e, S\_i | 2 | S\_e | S\_e, S\_i | S\_e,S\_i ∈ [0,1] | Deco et al. 2014 |
| `generic_2d_oscillator` | Generic2dOscillator | V, W | 2 | V | V, W | V∈[−2,4], W∈[−6,6] | FitzHugh 1961 |
| `kuramoto` | Kuramoto | θ | 1 | θ | θ | - | Kuramoto 1975 |
| `sup_hopf` | SupHopf | x, y | 2 | x, y | x, y | x,y ∈ [−5,5] | Deco et al. 2017 |
| `linear` | Linear | x | 1 | x | x | x ∈ [−1,1] | - |
| `larter_breakspear` | LarterBreakspear | V, W, Z | 3 | V | V | V,W,Z ∈ [−1.5,1.5] | Breakspear et al. 2003 |
| `coombes_byrne_2d` | CoombesByrne2D | r, V | 2 | r, V | r, V | r ≥ 0 | Coombes & Byrne 2019 |
| `gast_sd` | GastSchmidtKnosche\_SD | r, V, A, B | 4 | r, V | r, V | r ≥ 0 | Gast et al. 2020 |
| `gast_sf` | GastSchmidtKnosche\_SF | r, V, A, B | 4 | r, V | r, V | r ≥ 0 | Gast et al. 2020 |
| `vep` | VEP | x, y | 2 | x | x, y | - | Jirsa et al. 2014 |
| `ghb` | GHB | x, y | 2 | x, y | x, y | x,y ∈ [−5,5] | Deco et al. 2017 |
| `sl` | StuartLandau | x, y | 2 | x, y | x, y | x,y ∈ [−5,5] | Stuart & Landau 1944 |
| `damped_oscillator` | DampedOscillator | x, y | 2 | x | - | x,y ≥ 0 | Lotka 1925 |

---

## Model details

### MontbrioPopulationRate (`mpr`)

**Import:** `from vbi.simulator.models.mpr import mpr`

Exact mean-field reduction of a QIF (Quadratic Integrate-and-Fire) network via the
Ott-Antonsen ansatz. Produces a self-consistent description of firing rate `r` and
mean membrane potential `V` for an all-to-all coupled population with Lorentzian
heterogeneity.

**Equations:**

$$\dot{r} = \frac{1}{\tau}\left(\frac{\Delta}{\pi\tau} + 2Vr\right)$$

$$\dot{V} = \frac{1}{\tau}\left(V^2 - (\pi\tau r)^2 + \eta + J\tau r + I + c_r \cdot c_r^{\rm net} + c_v \cdot c_V^{\rm net}\right)$$

**Key parameters:** `tau`, `eta` (excitability), `J` (synaptic weight), `Delta`
(heterogeneity), `cr`/`cv` (coupling weights on r / V channels).

**Notes:**
- `r` has a hard lower bound of 0 (firing rate cannot be negative).
- `cr=1, cv=0` by default → only firing-rate coupling is active.
- TVB equivalent: `MontbrioPazoRoxin` in `infinite_theta.py`.

---

### JansenRit (`jansen_rit`)

**Import:** `from vbi.simulator.models.jansen_rit import jansen_rit`

6-dimensional second-order ODE model of a cortical column with three
populations: pyramidal cells (y0, y3), excitatory interneurons (y1, y4),
and inhibitory interneurons (y2, y5). Uses a sigmoid transfer function
`S(v) = 2ν_max / (1 + exp(r(v0 − v)))`.

**State variables:** `y0` (pyramidal soma), `y1` (exc dendritic),
`y2` (inh dendritic), `y3 = ẏ0`, `y4 = ẏ1` (noisy), `y5 = ẏ2`.

**Key parameters:** `A`, `B` (EPSP/IPSP amplitudes), `a`, `b` (time constants),
`J` (average synaptic contacts), `mu` (mean input firing rate).

**Notes:**
- Long-range coupling enters `y4` (excitatory dendritic current).
- Only `y4` receives additive noise.
- TVB default parameters are used verbatim.

---

### WilsonCowan (`wilson_cowan`)

**Import:** `from vbi.simulator.models.wilson_cowan import wilson_cowan`

Two-population mean-field model of excitatory (`E`) and inhibitory (`I`)
neural populations with sigmoidal transfer functions. Each population has
its own time constant, gain, and threshold.

**Equations (simplified):**

$$\tau_e \dot{E} = -E + (k_e - r_e E)\, c_e\!\left[\sigma(x_E)\right]$$

$$\tau_i \dot{I} = -I + (k_i - r_i I)\, c_i\!\left[\sigma(x_I)\right]$$

where $x_E = \alpha_e(c_{ee}E - c_{ei}I + P - \theta_e + c^{\rm net})$ and
$\sigma(x) = (1 + e^{-a(x-b)})^{-1}$.

**Key parameters:** `c_ee`, `c_ei`, `c_ie`, `c_ii` (local coupling
coefficients), `tau_e`, `tau_i`, `P`, `Q` (external inputs),
`shift_sigmoid` (1.0 for TVB-compatible shifted sigmoid, 0.0 for plain sigmoid).

**Notes:**
- Long-range coupling enters only the excitatory population (`E`).
- Both `E` and `I` are bounded to `[0, 1]` and receive noise.
- `shift_sigmoid=1.0` matches TVB default exactly.

---

### ReducedWongWang (`reduced_wong_wang`)

**Import:** `from vbi.simulator.models.reduced_wong_wang import reduced_wong_wang`

One-dimensional biophysically plausible mean-field model of a cortical area.
The single state variable `S` represents the average NMDA gating variable.
The sigmoidal transfer function is `H(x) = (ax − b) / (1 − e^{−d(ax−b)})`.

**Equation:**

$$\dot{S} = -\frac{S}{\tau_s} + (1-S)\,\gamma\, H\!\left(wJ_N S + I_o + J_N c^{\rm net}\right)$$

**Key parameters:** `w` (local recurrence), `J_N` (NMDA coupling), `I_o`
(external input), `tau_s`, `a`, `b`, `d`, `gamma`.

**Notes:**
- `S ∈ [0, 1]` enforced by bounds.
- Coupling enters via `J_N · c`, so the long-range synaptic drive scales with `J_N`.
- TVB equivalent: `ReducedWongWang` in `wong_wang.py`.

---

### ReducedWongWangExcInh (`wong_wang_exc_inh`)

**Import:** `from vbi.simulator.models.wong_wang_exc_inh import wong_wang_exc_inh`

Two-population extension of ReducedWongWang with explicit excitatory (`S_e`)
and inhibitory (`S_i`) NMDA/GABA gating variables. Long-range coupling acts
only on the excitatory population.

**Key parameters:** `w_p` (local E recurrence), `J_N`, `J_i`, `I_o`, `G`
(global coupling pre-applied by the simulator), `lamda` (inhibitory LRC scale),
`W_e`, `W_i` (external input weights).

**Notes:**
- `G` is extracted by the VBI simulator to pre-scale the coupling term `c`.
  Do **not** multiply by `G` again inside `dfun_str`.
- Both `S_e` and `S_i` are bounded to `[0, 1]`.
- TVB equivalent: `ReducedWongWangExcInh` in `wong_wang_exc_inh.py`.

---

### Generic2dOscillator (`generic_2d_oscillator`)

**Import:** `from vbi.simulator.models.generic_2d_oscillator import generic_2d_oscillator`

Highly configurable 2D model that can reproduce FitzHugh-Nagumo, Morris-Lecar,
and many other neuron/population dynamics by adjusting its nullcline parameters.
`V` is the fast variable; `W` is the slow recovery variable.

**Equations:**

$$\dot{V} = d\tau\left(\alpha W - fV^3 + eV^2 + gV + \gamma I + \gamma c^{\rm net}\right)$$

$$\dot{W} = \frac{d}{\tau}\left(a + bV + \kappa V^2 - \beta W\right)$$

**Key parameters:** `tau` (time-scale hierarchy), `a`, `b`, `c_coeff` (`κ`),
`d`, `e`, `f`, `g` (nullcline shape), `alpha`, `beta`, `gamma`.

**Notes:**
- TVB parameter `c` is renamed to `c_coeff` here to avoid collision with the
  `c = coupling[0]` alias injected by the dfun codegen.
- Default parameters give FitzHugh-Nagumo-like excitable dynamics.
  Set `a=2.0` for a limit cycle.
- TVB equivalent: `Generic2dOscillator` in `oscillator.py`.

---

### Kuramoto (`kuramoto`)

**Import:** `from vbi.simulator.models.kuramoto import kuramoto`

Minimal 1D phase oscillator model. Each node has a phase angle `θ` and a
natural frequency `ω`. Long-range coupling tends to synchronize phases across nodes.

**Equation:**

$$\dot{\theta} = \omega + c^{\rm net}$$

**Key parameters:** `omega` (natural angular frequency, rad/ms).

**Notes:**
- `theta` has no bounds - the phase wraps naturally.
- For the classic sinusoidal Kuramoto coupling use
  `CouplingSpec(kind="kuramoto")` (computes `sin(θ_j − θ_i)` sums).
  With `kind="linear"` the coupling is a weighted average of phases, which is
  the standard network-science approximation.
- TVB equivalent: `Kuramoto` in `oscillator.py`.

---

### SupHopf (`sup_hopf`)

**Import:** `from vbi.simulator.models.sup_hopf import sup_hopf`

Normal form of the supercritical Hopf bifurcation in Cartesian coordinates.
Widely used to model resting-state fMRI functional connectivity.

**Equations:**

$$\dot{x} = (a - x^2 - y^2)\,x - \omega y + c_x^{\rm net}$$

$$\dot{y} = (a - x^2 - y^2)\,y + \omega x + c_y^{\rm net}$$

**Key parameters:** `a` (bifurcation parameter; `a < 0` → fixed point,
`a > 0` → limit cycle), `omega` (angular frequency, rad/ms).

**Notes:**
- Uses two coupling channels: `c_x = coupling[0]` for the `x` variable and
  `c_y = coupling[1]` for `y`. Both must be provided by the connectivity.
- TVB equivalent: `SupHopf` in `oscillator.py`.

---

### Linear (`linear`)

**Import:** `from vbi.simulator.models.linear import linear`

Simplest possible 1D model: linear damping plus afferent coupling. Useful as a
baseline and for testing the simulation infrastructure.

**Equation:**

$$\dot{x} = \gamma x + c^{\rm net}$$

**Key parameters:** `gamma` (damping; must be `< 0` for stability at zero coupling).

---

### LarterBreakspear (`larter_breakspear`)

**Import:** `from vbi.simulator.models.larter_breakspear import larter_breakspear`

Modified Morris-Lecar model of a cortical column with three currents (Ca, K, Na)
and an inhibitory interneuron population. Exhibits fixed points, limit cycles,
and chaos depending on parameters.

**State variables:** `V` (membrane voltage), `W` (K recovery), `Z` (inhibitory population).

**Equations:**

$$\dot{V} = t_s \Big[-(g_{Ca} + (1-C)r_{\rm NMDA}\,a_{ee}\,Q_V + C\,r_{\rm NMDA}\,a_{ee}\,c^{\rm net})\,m_{Ca}(V-V_{Ca})$$

$$\quad - g_K W(V-V_K) - g_L(V-V_L) - (g_{Na}\,m_{Na} + (1-C)\,a_{ee}\,Q_V + C\,a_{ee}\,c^{\rm net})(V-V_{Na})$$

$$\quad - a_{ie}\,Z\,Q_Z + a_{ne}\,I_{\rm ext}\Big]$$

$$\dot{W} = t_s\,\phi\,(m_K - W) / \tau_K, \quad \dot{Z} = t_s\,b\,(a_{ni}\,I_{\rm ext} + a_{ei}\,V\,Q_V)$$

where $m_X = \tfrac{1}{2}(1 + \tanh((V-T_X)/\delta_X))$,
$Q_V = \tfrac{1}{2}Q_{V{\rm max}}(1+\tanh((V-V_T)/\delta_V))$,
$Q_Z = \tfrac{1}{2}Q_{Z{\rm max}}(1+\tanh((Z-Z_T)/\delta_Z))$.

**Key parameters:** `gCa`, `gK`, `gL`, `gNa`, `C` (LRC weight), `phi`,
`aee`, `aie`, `aei`, `d_V` (main bifurcation parameter).

**Notes:**
- All five intermediates (`m_Ca`, `m_Na`, `m_K`, `Q_V`, `Q_Z`) are inlined into the
  `dfun_str` expressions.
- **Dynamics switch:** `d_V < 0.55` → fixed point; `0.55–0.59` → limit cycle;
  `d_V > 0.59` → chaos.
- Only `V` receives additive noise.
- TVB equivalent: `LarterBreakspear` in `larter_breakspear.py`.

---

### CoombesByrne2D (`coombes_byrne_2d`)

**Import:** `from vbi.simulator.models.coombes_byrne_2d import coombes_byrne_2d`

Two-dimensional Ott-Antonsen mean-field reduction of an infinite all-to-all
network of QIF (theta) neurons. `r` is the average firing rate; `V` is the
average membrane potential.

**Equations** (with inline intermediate $g = k\pi r$):

$$\dot{r} = \frac{\Delta}{\pi} + 2Vr - k\pi r^2$$

$$\dot{V} = V^2 - \pi^2 r^2 + \eta + (v_{\rm syn} - V)\,k\pi r + c_r^{\rm net}$$

**Key parameters:** `Delta` (Lorentzian half-width), `eta` (mean excitability),
`k` (local conductance), `v_syn` (synaptic reversal potential).

**Notes:**
- `r ≥ 0` enforced by lower bound.
- Long-range coupling enters only the `V` equation.
- TVB equivalent: `CoombesByrne2D` in `infinite_theta.py`.

---

### GastSchmidtKnosche\_SD (`gast_sd`)

**Import:** `from vbi.simulator.models.gast_sd import gast_sd`

Four-dimensional QIF mean-field with **Synaptic Depression** adaptation.
`A` and `B` represent the adaptation variable and its derivative.

**Equations:**

$$\dot{r} = \frac{1}{\tau}\!\left(\frac{\Delta}{\pi\tau} + 2Vr\right)$$

$$\dot{V} = \frac{1}{\tau}\!\left(V^2 - \pi^2\tau^2 r^2 + \eta + J\tau r(1-A) + I + c_r c_r^{\rm net} + c_v c_V^{\rm net}\right)$$

$$\dot{A} = \frac{B}{\tau_A}, \quad \dot{B} = \frac{1}{\tau_A}(-2B - A + \alpha r)$$

**Key parameters:** `tau`, `tau_A`, `alpha`, `J`, `eta`, `Delta`, `cr`, `cv`.

**Notes:**
- The `(1 − A)` factor on `J` models synaptic resource depletion (depression).
- `r ≥ 0` enforced by lower bound.
- TVB's Heun integrator clamps `r` inside the predictor sub-step; VBI clamps
  after the corrector. Trajectories diverge when `r → 0`, so direct TVB
  numeric comparison is not available.
- TVB equivalent: `GastSchmidtKnosche_SD` in `infinite_theta.py`.

---

### GastSchmidtKnosche\_SF (`gast_sf`)

**Import:** `from vbi.simulator.models.gast_sf import gast_sf`

Four-dimensional QIF mean-field with **Spike-Frequency Adaptation**.
Structurally identical to `gast_sd` except the `V` equation uses
`J τ r − A` (additive subtraction) instead of `J τ r (1 − A)` (multiplicative).

**Equations:**

$$\dot{V} = \frac{1}{\tau}\!\left(V^2 - \pi^2\tau^2 r^2 + \eta + J\tau r - A + I + c_r c_r^{\rm net} + c_v c_V^{\rm net}\right)$$

(all other equations same as `gast_sd`)

**Key parameters:** `tau`, `tau_A`, `alpha` (default 10.0 here vs 0.5 in SD),
`J`, `eta`, `Delta`, `cr`, `cv`.

**Notes:**
- `alpha` default differs from `gast_sd` (10.0 vs 0.5).
- Same clamping caveat as `gast_sd`.
- TVB equivalent: `GastSchmidtKnosche_SF` in `infinite_theta.py`.

---

### VEP (`vep`)

**Import:** `from vbi.simulator.models.vep import vep`

Seizure-permittivity 2D model - a simplified Epileptor for virtual epileptic
patient (VEP) whole-brain simulations. `x` is the fast seizure-activity variable;
`y` is the slow permittivity variable.

**Equations:**

$$\dot{x} = 1 - x^3 - 2x^2 - y + I_{\rm ext}$$

$$\dot{y} = \frac{1}{\tau}\!\left(4(x - \eta) - y - G\!\sum_j W_{ij}(x_j - x_i)\right)$$

**Key parameters:** `tau` (slow time constant), `eta` (node excitability),
`iext` (external forcing), `G` (coupling strength).

**Notes:**
- Coupling is **Laplacian**: $G\sum_j W_{ij}(x_j - x_i)$.
  The coupling layer computes $c = G \cdot W x$, so the Laplacian becomes
  $c - G \cdot \text{row\_sum} \cdot x$ in `dfun_str`.
- **`row_sum` must be set** via `node_params`:
  ```python
  node_params={"row_sum": weights.sum(axis=1)}
  ```
- `eta` and `iext` are per-node heterogeneous parameters - pass via `node_params`.
- Reference: Jirsa VK et al. *Brain* 137(8):2210-2230, 2014.

---

### GHB (`ghb`)

**Import:** `from vbi.simulator.models.ghb import ghb`

Generic Hopf Bifurcation (Stuart-Landau) oscillator in Cartesian coordinates with
per-node bifurcation parameter `eta` and frequency `omega`. This is the neural
part of `GHB_sde`; BOLD output is obtained via `MonitorSpec(kind="bold")`.

**Equations:**

$$\dot{x} = (\eta - x^2 - y^2)\,x - \omega y + G\!\sum_j W_{ij}(x_j - x_i)$$

$$\dot{y} = (\eta - x^2 - y^2)\,y + \omega x + G\!\sum_j W_{ij}(y_j - y_i)$$

**Key parameters:** `eta` (bifurcation parameter per node), `omega` (angular
frequency per node, rad/ms), `G` (global coupling strength).

**Notes:**
- Coupling is **Laplacian** on both `x` and `y`.
  Uses two coupling channels: `c_x = G·Wx`, `c_y = G·Wy`.
  In `dfun_str`: `c_x - G*row_sum*x` and `c_y - G*row_sum*y`.
- **`row_sum` must be set** via `node_params`.
- `eta` and `omega` are per-node - pass via `node_params` for heterogeneous networks.
- Default `omega` ≈ 0.2513 rad/ms (40 Hz).
- Reference: Deco G et al. *Sci Rep* 7:3095, 2017.

---

### StuartLandau (`sl`)

**Import:** `from vbi.simulator.models.sl import sl`

Stuart-Landau oscillator in Cartesian form - mathematically identical to `ghb`
but with scalar (global) bifurcation parameter `a` and frequency `omega`.
Use `sl` when all nodes share the same intrinsic dynamics; use `ghb` for
heterogeneous networks.

**Equations:**

$$\dot{x} = (a - x^2 - y^2)\,x - \omega y + G\!\sum_j W_{ij}(x_j - x_i)$$

$$\dot{y} = (a - x^2 - y^2)\,y + \omega x + G\!\sum_j W_{ij}(y_j - y_i)$$

**Key parameters:** `a` (global bifurcation parameter), `omega` (global angular
frequency, rad/ms), `G` (global coupling strength).

**Notes:**
- Same Laplacian coupling pattern as `ghb`; `row_sum` must be set via `node_params`.
- `sup_hopf` uses the same equations but with **linear** (non-Laplacian) coupling.
  Choose `sl` when the reference model uses difference coupling; choose `sup_hopf`
  when the weights are already row-normalised or linear coupling is preferred.
- Default `omega` ≈ 0.2513 rad/ms (40 Hz); default `G` = 0.
- Reference: Stuart A & Landau L, 1944; Deco G et al. *Sci Rep* 7:3095, 2017.

---

### DampedOscillator (`damped_oscillator`)

**Import:** `from vbi.simulator.models.damped_oscillator import damped_oscillator`

Nonlinear 2D damped oscillator (Lotka-Volterra type). Exhibits fixed points,
limit cycles, and heteroclinic orbits depending on `a` and `b`. A single-node
model with no network coupling.

**Equations:**

$$\dot{x} = x - xy - ax^2$$

$$\dot{y} = xy - y - by^2$$

**Key parameters:** `a` (x-damping), `b` (y-damping).

**Notes:**
- No noise variables - deterministic only.
- Network coupling is not used; set `weights=np.zeros((N, N))` or run with `N=1`.
- `x, y ≥ 0` enforced by lower bounds.
- Reference: Lotka AJ, 1925; Volterra V, 1926.

---

## Usage

```python
from vbi.simulator import Simulator
from vbi.simulator.models import mpr, wilson_cowan, generic_2d_oscillator
from vbi.simulator.spec import SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec
import numpy as np

# Choose any model
model = generic_2d_oscillator

W = np.abs(np.random.default_rng(0).standard_normal((80, 80)))
np.fill_diagonal(W, 0.0)

spec = SimulationSpec(
    model=model,
    integrator=IntegratorSpec(method="heun", dt=0.01),
    coupling=CouplingSpec(kind="linear", a=0.05),
    monitors=(MonitorSpec(kind="tavg", period=1.0),),
    weights=W,
)

sim = Simulator(spec, backend="numba")   # or "numpy", "cpp"
t, data = sim.run(duration=5000.0)["tavg"]
# data shape: (n_steps, n_sv, n_nodes)
```

### Node-heterogeneous parameters

Any scalar parameter can be replaced with a per-node array:

```python
spec = SimulationSpec(
    model=mpr,
    ...,
    node_params={
        "eta": np.linspace(-5.5, -3.5, 80),   # different excitability per node
    },
)
```

### Parameter sweeps

```python
from vbi.simulator import Sweeper
from vbi.simulator.spec.sweep import SweepSpec

sweep_spec = SweepSpec(params={"eta": np.linspace(-5.5, -3.5, 100)})
sweeper = Sweeper(spec, sweep_spec, backend="numba")
labels, values = sweeper.run(duration=5000.0)
```

---

## Adding a new model

1. Create `vbi/simulator/models/my_model.py`.
2. Define a `ModelSpec` object:

```python
from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

my_model = ModelSpec(
    name="MyModel",
    state_variables=(
        StateVar("x", default_init=0.0, noise=True, lower_bound=None, upper_bound=None),
    ),
    parameters=(
        Parameter("a", 1.0, "description"),
    ),
    cvar=("x",),          # state vars that enter long-range coupling
    dfun_str={
        "x": "a * x + c", # 'c' = coupling[0]; 'c_x' also valid
    },
    noise_variables=("x",),
    reference="Author et al. Journal Year.",
)
```

3. Export it from `vbi/simulator/models/__init__.py`.
4. Add validation tests in `vbi/tests/validation/test_<name>_numpy.py`.

### dfun\_str rules

| Symbol | Meaning |
|--------|---------|
| `c` | Alias for `coupling[0]` (first cvar) |
| `c_<sv>` | Coupling for state var `<sv>`, e.g. `c_r`, `c_V` |
| `pi` | π (3.14159…) |
| `exp`, `log`, `sin`, `cos`, `tanh`, `sqrt`, `abs` | Standard math |
| `<param>` | Any parameter name from `ModelSpec.parameters` |
| `<sv>` | Any state variable name from `ModelSpec.state_variables` |

**Avoid** parameter names `c` and `c_<cvar_name>` - they are reserved for
coupling aliases. For example, if `cvar=("V",)` then `c_V` is reserved; rename
any parameter that would collide.

Intermediate variables are **not** supported - inline them directly in the
expression string (see `larter_breakspear.py` for an example with five inlined
intermediates).

### Laplacian coupling

Several models (VEP, GHB, SL) use difference coupling
$G \sum_j W_{ij}(x_j - x_i)$ rather than the default linear sum.
The framework coupling layer computes $c = G \cdot W x$, so the Laplacian
expands to $c - G \cdot \text{row\_sum}_i \cdot x_i$ inside `dfun_str`.
Implement this pattern by:

1. Adding a `row_sum` parameter to the `ModelSpec`:
   ```python
   Parameter("row_sum", 1.0, "per-node incoming weight sum")
   ```
2. Writing the corrected coupling term in `dfun_str`:
   ```python
   "x": "... + c - G*row_sum*x"
   ```
3. Setting `node_params` at simulation time:
   ```python
   node_params={"row_sum": weights.sum(axis=1)}
   ```
