# Inference Engine Milestones — `vbi/inference/`

> **Goal:** A torch-free, high-performance SBI engine with a sbi-compatible
> API.  Backends progress: numpy/autograd → numba → JAX.  The existing
> `vbi/cde.py` is wrapped and gradually replaced; users never need to change
> their code.

---

## Package target structure

```
vbi/
  inference/
    __init__.py          # public: SNPE, BoxUniform, Gaussian, Posterior
    _api.py              # SNPE / SNLE classes (sbi-compatible interface)
    _prior.py            # BoxUniform, Gaussian, CustomPrior
    _posterior.py        # Posterior object
    _estimators/
      __init__.py
      base.py            # ConditionalDensityEstimator ABC (from cde.py)
      mdn.py             # MDNEstimator (from cde.py)
      maf.py             # MAFEstimator (from cde.py)
    _backends/
      __init__.py
      numpy_.py          # autograd/numpy (current, default)
      numba_.py          # Numba JIT forward pass (MI-numba)
      jax_.py            # JAX: jit + vmap + grad (MI1)
  cde.py                 # deprecated shim → re-exports from vbi.inference
```

**Backend selector** (transparent to user):
```python
SNPE(prior, density_estimator='maf', backend='auto')
# 'auto' → jax if installed, else numpy
# 'numpy' → autograd/numpy (always available)
# 'numba' → Numba JIT (install vbi[numba])
# 'jax'   → JAX XLA (install vbi[jax])
```

---

## API Design Decision: mirror `sbi`

**Decision: yes, mirror the `sbi` API at the high level.**

The `sbi` API is well-designed and already familiar to computational
neuroscience users.  Mirroring it means:

- Users can switch from `sbi` to `vbi.cde` by changing one import and
  converting `torch.Tensor` → `np.ndarray`.
- All `sbi` tutorials and examples translate almost verbatim.
- `vbi.sbi_inference.Inference` (the current heavy wrapper) and the new
  light CDE stack share the same mental model.

### Two-layer architecture

```
┌──────────────────────────────────────────────────────────┐
│  HIGH-LEVEL  sbi-compatible API  (vbi/cde_api.py)        │
│  SNPE, SNLE  ←→  append_simulations → train →            │
│              build_posterior → posterior.sample(...)      │
│                                                          │
│  Prior objects: BoxUniform, Gaussian, CustomPrior        │
│  Posterior object: .sample((n,), x=x_obs)                │
│                    .log_prob(theta, x=x_obs)              │
│                    .map(x=x_obs)                          │
└──────────────────┬───────────────────────────────────────┘
                   │ delegates to
┌──────────────────▼───────────────────────────────────────┐
│  LOW-LEVEL  estimator building blocks  (vbi/cde.py)      │
│  MDNEstimator, MAFEstimator                              │
│  ConditionalDensityEstimator ABC                         │
│  backends: autograd (default) → JAX (MI1) → torch (MI5) │
└──────────────────────────────────────────────────────────┘
```

### API comparison: sbi 0.26 vs vbi.cde (target)

Verified against sbi 0.26.1 exact signatures.

| `sbi` call | `vbi.cde` call | Notes |
|-----------|---------------|-------|
| `from sbi.inference import SNPE` | `from vbi.cde import SNPE` | import only |
| `BoxUniform(low=torch.tensor([0.]), high=torch.tensor([1.]))` | `BoxUniform(low=np.array([0.]), high=np.array([1.]))` | torch → numpy |
| `SNPE(prior, density_estimator='maf'\|'mdn'\|'nsf', device='cpu')` | `SNPE(prior, density_estimator='maf'\|'mdn', backend='auto')` | `nsf` deferred to MI1+; `backend` is new |
| `inference.append_simulations(theta, x, proposal=None)` | `inference.append_simulations(theta, x, proposal=None)` | ✅ identical |
| `inference.train(training_batch_size=200, learning_rate=5e-4, validation_fraction=0.1, stop_after_epochs=20, clip_max_norm=5.0, ...)` | `inference.train(training_batch_size=200, learning_rate=5e-4, validation_fraction=0.1, stop_after_epochs=20, clip_max_norm=5.0, ...)` | ✅ identical kwargs (mapped internally) |
| `inference.build_posterior(estimator, sample_with='direct'\|'mcmc'\|'vi')` | `inference.build_posterior(estimator, sample_with='direct'\|'mcmc')` | `vi` deferred to MI4+ |
| `posterior.sample((1000,), x=x_obs)` | `posterior.sample((1000,), x=x_obs)` | ✅ identical (tuple shape) |
| `posterior.log_prob(theta, x=x_obs)` | `posterior.log_prob(theta, x=x_obs)` | ✅ identical — note **theta first**, x second |
| `posterior.map(x=x_obs)` | `posterior.map(x=x_obs)` | ✅ identical |
| `posterior.set_default_x(x_obs)` | `posterior.set_default_x(x_obs)` | ✅ identical |
| `prior.sample((n,))` | `prior.sample((n,))` | ✅ identical |

**Critical difference from current `vbi.cde` low-level API:**
`MDNEstimator.log_prob(features, params)` has features **first**.
`posterior.log_prob(theta, x=x_obs)` has params **first** (sbi convention).
The sbi-compatible layer swaps the order; the low-level API stays as-is.

**`train()` kwargs that map directly** (MAFEstimator already has them):

| sbi kwarg | vbi.cde internal kwarg | Status |
|-----------|----------------------|--------|
| `training_batch_size=200` | `batch_size` | ❌ add in MI0 |
| `learning_rate=5e-4` | `learning_rate` | ✅ exists |
| `validation_fraction=0.1` | `validation_fraction` | ✅ exists |
| `stop_after_epochs=20` | `stop_after_epochs` | ✅ exists |
| `clip_max_norm=5.0` | `clip_max_norm` | ✅ exists |
| `max_num_epochs` | `n_iter` | ✅ rename in adapter |
| `num_atoms=10` | N/A (SNPE-C specific) | accept + ignore in MI-API; implement in MI3 |

**Migration from `sbi` to `vbi.cde` is a 2-line change:**
```python
# Before
from sbi.inference import SNPE
from sbi.utils import BoxUniform
theta, x = torch.tensor(theta_np), torch.tensor(x_np)

# After
from vbi.cde import SNPE
from vbi.cde import BoxUniform
theta, x = theta_np, x_np   # plain numpy — no conversion needed
```

### Gradual migration plan for existing `vbi/cde.py`

The existing `MDNEstimator.train(theta, x)` / `MDNEstimator.sample(x_obs, n_samples, rng)`
API is **kept as the low-level interface** and is not removed.
The sbi-compatible layer wraps it:

```
Phase 1 (MI-API):   Add SNPE/SNLE/prior/Posterior in vbi/cde_api.py
                    Internally delegates to MDNEstimator/MAFEstimator
                    Old low-level API unchanged

Phase 2 (MI1+):     JAX backend added; low-level API gains backend= kwarg
                    SNPE auto-selects JAX when available

Phase 3 (MI3+):     Sequential rounds (append_simulations multi-round)
                    Old single-call train() deprecated (not removed)

Phase 4 (future):   Optional: soft-deprecate MDNEstimator.train() in favour
                    of SNPE; remove only after 2 release cycles
```

---

## Motivation

| Issue | Current | Target |
|-------|---------|--------|
| Heavy dependencies | `sbi` requires `torch` (1+ GB) + `pyknos` | Pure NumPy / JAX only |
| GPU training | torch.cuda required | JAX automatic via XLA |
| Mini-batch training | Full-batch only | Stochastic mini-batches |
| Sequential SBI | Not in CDE | SNPE-C style rounds |
| Differentiable through inference | Not possible (autograd CPU only) | JAX end-to-end |
| Model serialization | No save/load | `save(path)` / `load(path)` |
| Prior support | None in CDE | BoxUniform, Gaussian, custom |
| Posterior sampling quality | Basic rejection | MCMC + prior truncation |

---

## Current state (May 2026)

### What exists in `vbi/cde.py`

| Component | Status | Notes |
|-----------|--------|-------|
| `ConditionalDensityEstimator` ABC | ✅ Complete | Adam optimizer, early stopping, val split, grad clip |
| `MDNEstimator` | ✅ Complete | Gaussian mixture, full covariance, MLP backbone, rejection sampling |
| `MAFEstimator` | ✅ Complete | MADE blocks, ActNorm, permutations, z-scoring, PCA embedding |
| `MAFEstimator0` | ⚠️ Legacy | Original simpler MAF, kept for backward compat |
| Backend | `autograd` (CPU only) | Automatic diff via `autograd.numpy` |
| Training | Full-batch | No mini-batching → slow on large N |
| Save/load | ❌ Missing | Trained weights not serializable |
| Prior integration | ❌ Missing | No prior, no posterior truncation |
| Sequential rounds | ❌ Missing | No SNPE-C / round-by-round |
| MCMC sampling | ❌ Missing | Only direct ancestral sampling |

### What exists in `vbi/sbi_inference.py`

- Thin wrapper around `sbi` (SNPE, SNLE, SNRE)
- Requires `torch` + `sbi` — works well but heavy
- Will remain as the "heavy" option; CDE path is the "light" option

---

## ★ FIRST STEP: MI-API — sbi-compatible API with numpy backend

> This is the concrete starting point.  No new math, no new backends.
> The goal is to get the right API shape working end-to-end so every
> subsequent milestone just plugs in a faster backend underneath.

### What "done" looks like

```python
from vbi.inference import SNPE, BoxUniform
import numpy as np

prior = BoxUniform(low=np.array([0., -5.]), high=np.array([2., 0.]))

inference = SNPE(prior=prior, density_estimator='maf')
inference = inference.append_simulations(theta, x)   # theta, x: np.ndarray
estimator = inference.train(
    training_batch_size=256,
    learning_rate=5e-4,
    stop_after_epochs=20,
)
posterior = inference.build_posterior(estimator)

samples  = posterior.sample((1000,), x=x_obs)        # sbi signature
log_probs = posterior.log_prob(theta, x=x_obs)        # params first, like sbi
```

Users who currently use `sbi` can switch by changing the import and
dropping `torch.tensor()` calls.  Everything else is identical.

---

## MI-API — sbi-compatible high-level API  *(do this alongside MI0)*

**Goal:** Create `vbi/cde_api.py` that mirrors the `sbi` interface exactly,
wrapping the existing `MDNEstimator`/`MAFEstimator` as its backend.
Users switching from `sbi` need no mental-model change.

**Files to create / move:**

```
vbi/inference/__init__.py         NEW — public exports
vbi/inference/_prior.py           NEW — BoxUniform, Gaussian, CustomPrior
vbi/inference/_posterior.py       NEW — Posterior object
vbi/inference/_api.py             NEW — SNPE class (sbi-compatible)
vbi/inference/_estimators/base.py MOVE from vbi/cde.py (ConditionalDensityEstimator)
vbi/inference/_estimators/mdn.py  MOVE from vbi/cde.py (MDNEstimator)
vbi/inference/_estimators/maf.py  MOVE from vbi/cde.py (MAFEstimator)
vbi/inference/_backends/numpy_.py NEW — autograd backend (wraps existing code)
vbi/cde.py                        KEEP as deprecated shim
```

**Tasks:**

- [ ] **Create `vbi/inference/` package** — move estimator classes from
      `cde.py` into `_estimators/`; keep `vbi/cde.py` as a re-export shim
      with `DeprecationWarning`
- [ ] **`BoxUniform(low, high)`** — accepts numpy/list;
      `.sample((n,))` → `(n, d)`; `.log_prob(theta)` → 0 inside, `-inf` outside
- [ ] **`Gaussian(mean, std)`** — `.sample((n,))` + `.log_prob(theta)`
- [ ] **`CustomPrior(sample_fn, log_prob_fn)`** — user-supplied callables
- [ ] **`Posterior` class** — wraps the trained estimator:
      ```python
      posterior.sample((1000,), x=x_obs)    # tuple shape, like sbi
      posterior.log_prob(theta, x=x_obs)    # params FIRST, x as kwarg
      posterior.map(x=x_obs)                # gradient ascent on log_prob
      posterior.set_default_x(x_obs)        # store default observation
      ```
      Note: `log_prob(theta, x)` internally calls `estimator.log_prob(x, theta)`
      — the argument swap is handled here so the low-level API is unchanged.
- [ ] **`SNPE(prior, density_estimator='maf'|'mdn', backend='numpy')`**:
      - `append_simulations(theta, x, proposal=None)` — stores data per round
      - `train(training_batch_size=256, learning_rate=5e-4,
               validation_fraction=0.1, stop_after_epochs=20,
               clip_max_norm=5.0, max_num_epochs=2000, **kwargs)`
        maps sbi kwarg names → internal estimator kwarg names
      - `build_posterior(density_estimator=None, sample_with='direct')` →
        `Posterior`
      - `num_atoms` accepted but ignored (implement in MI3)
- [ ] **`SNLE`** — stub that raises `NotImplementedError` with message
      "SNLE coming in a future release; use SNPE for now"
- [ ] **Mini-batch support** in `MAFEstimator.train()` (`batch_size` kwarg,
      shuffle indices each epoch) — needed for `training_batch_size` to work
- [ ] **Drop-in migration test**:
      ```python
      # Assert this works with vbi.inference.SNPE
      inference = SNPE(prior=prior, density_estimator='maf')
      inference.append_simulations(theta_np, x_np)
      est = inference.train(training_batch_size=128, stop_after_epochs=10)
      post = inference.build_posterior(est)
      s = post.sample((500,), x=x_obs_np)
      assert s.shape == (500, n_params)
      ```
- [ ] `vbi/cde.py` becomes:
      ```python
      import warnings
      warnings.warn("vbi.cde is deprecated; use vbi.inference", DeprecationWarning)
      from vbi.inference._estimators.mdn import MDNEstimator
      from vbi.inference._estimators.maf import MAFEstimator, MAFEstimator0
      from vbi.inference._estimators.base import ConditionalDensityEstimator
      ```

**Effort:** 3-4 days.  Math already exists; this is package restructuring +
API glue + argument mapping.

---

## MI0 — CDE hardening (quick wins, no new backend)

**Goal:** Fix correctness issues and add missing basics to the existing
autograd implementation before adding new backends.

**Tasks:**

- [ ] **Mini-batch training**: add `batch_size` parameter to `train()`;
      shuffle indices each epoch; default `batch_size=256`
- [ ] **Save / load**: `estimator.save(path)` → pickle or npz;
      `MDNEstimator.load(path)` → class method restoring weights + config
- [ ] **Logging cleanup**: replace `print(...)` with `logging.getLogger(...)`;
      verbose level controlled by `verbose: bool = False`
- [ ] **Numerical stability**:
      - `logsumexp` already used in MDN; check MAF log_det underflow
      - Clip log_sigma more aggressively in MAFEstimator when needed
- [ ] **API consistency**: `train()` signature unified across MDN/MAF
      (currently MAF has extra params not in base); lift common params to ABC
- [ ] **Test coverage**: `test_cde.py` covers basic MDN; add
      - MAF round-trip: `log_prob` + `sample` consistency test
      - Save/load: weights survive serialization
      - Mini-batch: loss converges same as full-batch (loose tolerance)

**Effort:** Small, 1–2 days.

---

## MI-numba — Numba JIT backend

**Goal:** 5–10× faster training on CPU with no GPU and no JAX dependency.
The forward pass and loss function are the bottleneck for small–medium N;
JIT-compiling them with Numba gives immediate speed without changing the API.

**Design:** Only the inner hot loops are JIT-compiled.  The Python training
loop (Adam, validation, early stopping) stays in Python.

**Tasks:**

- [ ] `vbi/inference/_backends/numba_.py`:
      - `@njit` MLP forward pass for MDN: `mdn_forward(weights_flat, features)`
      - `@njit` MDN log-likelihood: `mdn_log_prob(alpha, mu, L_prec, params)`
      - `@njit` MAF forward pass: `maf_forward(weights_flat, features, params)`
      - Flat weight vector ↔ dict conversion utilities
- [ ] Gradient computation via finite differences for the Numba path
      (not auto-diff; acceptable for moderate param counts < 10k weights)
      OR use `autograd` for gradient but Numba for forward (hybrid)
- [ ] Numba backend selected via `SNPE(..., backend='numba')`
- [ ] Benchmark: autograd vs numba on N=1k, N=10k, N=100k
- [ ] Tests: Numba and numpy backends give same loss to rtol=1e-4

**Effort:** Medium, 3–5 days.

---

## MI1 — JAX backend for CDE

**Goal:** Replace `autograd.numpy` with JAX as the computation backend,
keeping the same Python API. JAX gives: JIT compilation, GPU/TPU via XLA,
`vmap` for batch evaluation, and `jax.grad` for differentiability.

**Design principle:** The backend is a hidden implementation detail.
Users never import JAX directly; the estimator selects it automatically.

```python
# Same API regardless of backend
mdn = MDNEstimator(n_components=8, hidden_sizes=(64, 64))
mdn.train(theta, x)                    # JIT-compiled under the hood
samples = mdn.sample(x_obs, n_samples=1000, rng=rng)
```

**Implementation strategy:**

The current `autograd.numpy` code is structurally compatible with JAX.
Key differences to handle:

| `autograd` pattern | JAX equivalent |
|--------------------|----------------|
| `anp.asarray(x)` | `jnp.asarray(x)` |
| `from autograd import grad` | `from jax import grad, jit, vmap` |
| `anp.random.RandomState(seed)` | `jax.random.PRNGKey(seed)` + functional PRNG |
| In-place weight update `w[k] -= lr * g` | Immutable update: `w = {k: w[k] - lr*g for k}` |
| `anp.linalg.inv` in MDN sample | `jnp.linalg.inv` |
| Full-batch gradient | `jax.jit(grad(loss))` + mini-batch loop |

**Tasks:**

- [ ] Create `vbi/cde_jax.py` (or add `backend='jax'` dispatch in `cde.py`)
- [ ] Functional Adam optimizer (pure functions, no mutation):
      ```python
      def adam_update(weights, grads, m, v, t, lr): ...
      ```
- [ ] JIT-compile training loop: `jax.jit(grad(loss_fn))`
- [ ] `jax.random` PRNG threading through sample/train
- [ ] `vmap`-compatible `log_prob` and `sample` for batch conditions
- [ ] `jax.grad` through the full `log_prob` (for gradient-based posterior
      exploration)
- [ ] Mini-batch training with `jnp.take`
- [ ] Data normalization inside JIT-safe transforms
- [ ] Benchmark: autograd vs JAX on N=10k, N=100k, CPU, GPU
- [ ] Tests: numerical match autograd vs JAX (rtol=1e-4)
- [ ] Fallback: if JAX not installed, silently use autograd

**Effort:** Medium, 3–5 days. Most logic is a direct translation.

---

## MI2 — Prior support and posterior API

**Goal:** Match `sbi`'s prior handling so users can seamlessly switch between
`sbi_inference.Inference` and `cde.MDNEstimator` / `cde.MAFEstimator`.

**Tasks:**

- [ ] `BoxUniformPrior(low, high)` — uniform on a hyperrectangle
- [ ] `GaussianPrior(mean, std)` — independent Gaussians
- [ ] `CustomPrior(log_prob_fn, sample_fn)` — user-supplied callables
- [ ] `Prior.sample(n)` → `(n, d)` array
- [ ] `Prior.log_prob(theta)` → `(n,)` array
- [ ] Integrate prior into estimator: `log_posterior = log_prob_x_given_theta + log_prior`
- [ ] `estimator.build_posterior(prior, x_obs)` → `Posterior` object with:
      - `posterior.sample(n_samples)` — draws from approx posterior
      - `posterior.log_prob(theta)` — evaluates log posterior
      - `posterior.map()` — maximum a posteriori estimate
- [ ] Prior truncation in MDN sampler (automatic rejection based on prior bounds)
- [ ] `test_cde_estimation.py` — add prior-weighted sampling tests

**Effort:** Medium, 2–3 days.

---

## MI3 — Sequential Neural Posterior Estimation (SNPE-C style)

**Goal:** Implement multi-round SBI where each round's posterior narrows the
simulation budget by focusing on high-posterior regions.

**Design** (matches sbi's SNPE-C / APT):

```python
estimator = MAFEstimator(n_flows=4)

# Round 1: simulate from prior
theta_r1, x_r1 = simulator.run_batch(prior.sample(1000))
estimator.append_simulations(theta_r1, x_r1)
posterior_r1 = estimator.train().build_posterior(prior, x_obs)

# Round 2: simulate from round-1 posterior
theta_r2, x_r2 = simulator.run_batch(posterior_r1.sample(500))
estimator.append_simulations(theta_r2, x_r2)
posterior_r2 = estimator.train().build_posterior(prior, x_obs)
```

**Tasks:**

- [ ] `estimator.append_simulations(theta, x)` — accumulate rounds
- [ ] Proposal distribution tracking: store round-specific `proposal` per sample
- [ ] **Atomic Posterior Transform (APT)** loss:
      `log p(θ|x) - log Z(x)` with importance weights for proposal correction
- [ ] Round management: `estimator.round_` counter, per-round diagnostics
- [ ] `train(proposal=prior_or_posterior)` — optional proposal argument for
      importance-weighted NLL
- [ ] Simulation wrapper: `SequentialSimulator(spec, pipeline, prior)` that
      wraps `vbi.Sweeper` with built-in round management
- [ ] Tests: 2-round SNPE recovers known posterior for linear-Gaussian model
- [ ] Example notebook: sequential inference for MPR G-parameter

**Effort:** Large, 1–2 weeks. Core math is straightforward; main work is
testing convergence.

---

## MI4 — MCMC posterior sampling

**Goal:** Use the trained density estimator as a likelihood surrogate inside
MCMC for high-quality posterior samples, especially when the neural posterior
has mode artifacts.

**Tasks:**

- [ ] `MCMCSampler(estimator, prior)` base class
- [ ] `MetropolisHastings(estimator, prior, step_size)`:
      - Proposal: random walk Gaussian
      - Accept/reject using `log_posterior = estimator.log_prob + prior.log_prob`
- [ ] `NUTS(estimator, prior)` (JAX backend only):
      - Uses `jax.grad(log_posterior)` for HMC leapfrog
      - Requires MI1 (JAX backend) + MI2 (prior)
- [ ] Convergence diagnostics: `R_hat`, effective sample size
- [ ] `sampler.run(x_obs, n_samples, n_warmup)` → `(n_samples, param_dim)`
- [ ] Tests: MCMC samples match analytical posterior for Gaussian model

**Effort:** Medium (MH) to Large (NUTS). MH is 1–2 days; NUTS requires
JAX backend (MI1) first.

---

## MI5 — PyTorch backend (optional, parity with sbi)

**Goal:** Allow the CDE estimators to run on PyTorch (GPU via CUDA, larger
models, familiar ecosystem) without requiring the full `sbi` library.

**Design:** Same ABC interface, PyTorch nn.Module underneath.

```python
mdn = MDNEstimator(n_components=8, hidden_sizes=(128, 128), backend='torch')
mdn.train(theta, x, device='cuda')
```

**Tasks:**

- [ ] `vbi/cde_torch.py` — PyTorch implementation of MDN and MAF
- [ ] `torch.nn.Module`-based forward pass (use PyTorch autograd)
- [ ] `torch.optim.Adam` training loop with DataLoader for mini-batches
- [ ] GPU support: `.to(device)` pattern
- [ ] Consistent `sample()` and `log_prob()` API matching autograd/JAX versions
- [ ] `estimator.to_numpy()` — convert PyTorch weights back to NumPy for
      serialization/interop
- [ ] Optional import: `backend='torch'` only if `torch` is installed
- [ ] Benchmark: PyTorch GPU vs JAX GPU vs autograd CPU

**Effort:** Medium, 3–5 days. Much of the architecture logic already exists.

---

## MI6 — End-to-end VBI integration

**Goal:** Seamless workflow from VBI simulator → features → posterior.

```python
from vbi.simulator import Sweeper
from vbi.cde import MAFEstimator, BoxUniformPrior
from vbi.inference_api import VBIInference

# Fully integrated — single API, no manual glue
inf = VBIInference(
    spec=sim_spec,
    pipeline=feature_pipeline,
    estimator=MAFEstimator(n_flows=4, backend='jax'),
    prior=BoxUniformPrior(low=[0.0, -5.0], high=[2.0, 0.0]),
)

# Round 1
inf.simulate(n_sims=2000, backend='numba')
inf.train()

# Round 2 (simulate from current posterior)
inf.simulate(n_sims=1000, round=2)
inf.train()

# Posterior
posterior = inf.build_posterior(x_obs=observed_features)
samples = posterior.sample(5000)
```

**Tasks:**

- [ ] `VBIInference` class in `vbi/inference_api.py`
- [ ] `inf.simulate(n_sims, backend)` → runs `Sweeper`, extracts features,
      stores `(theta, x)` internally
- [ ] `inf.train()` → delegates to chosen CDE estimator
- [ ] `inf.build_posterior(x_obs)` → `Posterior` object (MI2)
- [ ] Checkpointing: `inf.save(path)` / `inf.load(path)` — saves simulator
      spec, pipeline, estimator weights, round history
- [ ] `VBIInference.from_config(yaml_path)` — reproducible pipelines
- [ ] Diagnostic plots: `inf.plot_loss()`, `inf.pairplot()`, `inf.coverage()`
- [ ] JAX end-to-end: when `backend='jax'` + `estimator backend='jax'`,
      `jax.grad` flows through simulation → features → log_prob
- [ ] Example notebooks:
      - 1-round SNPE: MPR G parameter from FC
      - 2-round SNPE: VEP excitability from FCD
      - NUTS posterior refinement on real data

**Effort:** Large, 1–2 weeks for the API; notebooks add another week.

---

## MI7 — Performance and scalability

**Goal:** Handle N = 100k–1M simulation samples efficiently.

**Tasks:**

- [ ] **Mini-batch JAX**: `jax.lax.scan` over batches inside `jax.jit`;
      avoid Python loop overhead
- [ ] **Data pipeline**: lazy loading from disk when N > memory (`numpy.memmap`)
- [ ] **Distributed training** (optional): `jax.pmap` across multiple GPUs
- [ ] **Mixed precision**: float16 training with float32 loss accumulation
      (JAX and PyTorch backends)
- [ ] **Profiling tools**: `inf.benchmark(n_sims, n_iter)` → timing table
      comparing backends

**Effort:** Medium per feature, ongoing.

---

## Backend roadmap

| Milestone | Backend | What it enables | Dependency |
|-----------|---------|----------------|-----------|
| **MI-API** | numpy/autograd | sbi-compatible API, full workflow | `numpy`, `autograd` |
| **MI0** | numpy/autograd | mini-batch, save/load, logging | same |
| **MI-numba** | Numba JIT | 5–10× faster training on CPU, no GPU needed | `numba` |
| **MI1** | JAX | GPU, vmap, jit, gradient through posterior | `jax` |
| **MI2** | all | prior integration, Posterior object (part of MI-API) | MI-API |
| **MI3** | all | sequential rounds, multi-round SNPE | MI-API + MI2 |
| **MI4** | numpy + JAX | MCMC: MH (numpy), NUTS (JAX) | MI1 + MI2 |
| **MI6** | all | VBIInference end-to-end integration | MI1 + MI2 + MI3 |
| **MI5** | torch (optional) | large models, PyTorch ecosystem | MI-API |

## Priority order

```
★ STEP 1: MI-API  package restructure + sbi-compatible interface (numpy backend)
           → also covers MI0 mini-batch + save/load

  STEP 2: MI-numba  JIT-compile forward pass + loss
           (fastest return before JAX: 5-10× speedup, no new dependency)

  STEP 3: MI1 (JAX backend)
           → GPU, vmap, jit, full gradient path

  STEP 4: MI3 (sequential rounds)
           → multi-round SNPE-C

  STEP 5: MI4 (MCMC) — needs JAX for NUTS

  STEP 6: MI6 (end-to-end VBIInference)

  OPTIONAL: MI5 (torch backend) — only if torch already present
```

### Dependency graph

```
MI-API+MI0
  │
  ├──→ MI-numba  (fast CPU, independent of JAX)
  │
  ├──→ MI1 (JAX) ──→ MI4 (MCMC/NUTS)
  │                        │
  ├──→ MI2 (prior) ────────┤
  │                        │
  └──→ MI3 (sequential) ───┴──→ MI6 (end-to-end)

           MI5 (torch, optional) ─────────────────┘
```

---

## Comparison with `sbi`

| Feature | `sbi` | `vbi.cde` (target) |
|---------|-------|--------------------|
| MDN | ✅ | ✅ (done) |
| MAF | ✅ | ✅ (done) |
| NSF (neural spline flow) | ✅ | ❌ MI1+ |
| Sequential rounds (SNPE-C) | ✅ | ❌ MI3 |
| Prior support | ✅ | ❌ MI2 |
| MCMC refinement | ✅ (NUTS) | ❌ MI4 |
| GPU training | ✅ (torch.cuda) | ❌ MI1 (JAX) |
| vmap batch eval | ✅ | ❌ MI1 |
| Gradient through posterior | Limited | ✅ MI1 (JAX, full) |
| Mini-batch training | ✅ | ❌ MI0 |
| Save / load | ✅ | ❌ MI0 |
| Dependency size | ~2 GB | ~50 MB (numpy+jax) |
| Installation | `pip install sbi` | `pip install vbi` |
| Integration with VBI sim | ❌ manual | ✅ MI6 (native) |
| Numpy CDE speed (small N) | Slower | ✅ already faster |

---

## Open design questions

1. **Autograd vs JAX as default**: After MI1, should JAX replace autograd as
   the default backend? Autograd is pure Python (easier to debug); JAX is
   faster. Proposal: keep autograd as `backend='autograd'` fallback, JAX as
   default when installed.

2. **NSF (Neural Spline Flow)**: sbi's best-performing estimator. Requires
   rational-quadratic spline transforms. Doable in JAX, harder in autograd.
   Should this be MI1.5 or a separate milestone?

3. **`MAFEstimator0` deprecation**: The original simpler MAF should be
   deprecated in MI0 and removed in MI1. Confirm with user.

4. **Embedding network**: sbi supports a learned summary network
   `embedding_net` that maps raw data to features before conditioning.
   `MAFEstimator` already has PCA; should a learned MLP embedding be added?

5. **Amortized vs sequential**: the current CDE is fully amortized
   (one network for all conditions). Sequential SBI breaks amortization.
   Keep both paths or focus on amortized?
