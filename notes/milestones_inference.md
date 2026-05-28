# Inference Engine Milestones — `vbi/inference/`

> **Goal:** A torch-free, high-performance SBI engine with a sbi-compatible
> API.  Backends progress: numpy/autograd → JAX.  The existing
> `vbi/cde.py` is wrapped and gradually replaced; users never need to change
> their code.

---

## Feature comparison: sbi 0.26 vs vbi.inference (current + planned)

> Verified against sbi 0.26.1.  ✅ = done  ⚠️ = partial  ❌ = missing
> Priority: 🔴 high (blocks real use)  🟡 medium  🟢 low / defer

### Inference algorithms

| Feature | sbi | vbi.inference | Priority | Milestone |
|---------|-----|---------------|----------|-----------|
| SNPE / NPE-C | ✅ | ✅ | — | done |
| SNPE-A / SNPE-B (older) | ✅ | ❌ | 🟢 | not planned |
| SNLE (neural likelihood) | ✅ | ❌ stub | 🟡 | MI-SNLE |
| SNRE-A/B/C (neural ratio) | ✅ | ❌ | 🟢 | future |
| FMPE (flow matching) | ✅ | ❌ | 🟢 | future |
| MNPE / MNLE (mixed data) | ✅ | ❌ | 🟢 | future |
| SMC-ABC / MC-ABC | ✅ | ❌ | 🟢 | future |

### Density estimators

| Feature | sbi | vbi.inference | Priority | Milestone |
|---------|-----|---------------|----------|-----------|
| MAF | ✅ | ✅ | — | done |
| MDN | ✅ | ✅ | — | done |
| NSF (neural spline flow) | ✅ | ✅ | — | done |
| MADE (standalone) | ✅ | ❌ | 🟡 | future |
| Custom nn via callable | ✅ | ❌ | 🟡 | MI-custom-net |
| Embedding / summary network | ✅ | ✅ (EmbeddingNet) | — | done |
| z-score theta | ✅ | ✅ (MAF) | — | done |
| z-score x | ✅ | ✅ (MAF) | — | done |
| structured z-scoring | ✅ | ❌ | 🟢 | future |

### SNPE training features

| Feature | sbi | vbi.inference | Priority | Milestone |
|---------|-----|---------------|----------|-----------|
| Mini-batch training | ✅ | ✅ | — | done |
| Train / val split | ✅ | ✅ | — | done |
| Early stopping | ✅ | ✅ | — | done |
| Gradient clipping | ✅ | ✅ | — | done |
| Sequential rounds: append_simulations | ✅ | ✅ (data only) | ⚠️ | done |
| APT importance weights (num_atoms) | ✅ | ✅ | — | done (MI3) |
| Warm start (resume_training) | ✅ | ✅ | — | done |
| get_simulations() | ✅ | ✅ | — | done |
| Custom DataLoader kwargs | ✅ | ❌ | 🟢 | future |
| Discard prior samples (round weighting) | ✅ | ❌ | 🟡 | MI3 |

### Posterior sampling

| Feature | sbi | vbi.inference | Priority | Milestone |
|---------|-----|---------------|----------|-----------|
| Direct / ancestral sampling | ✅ | ✅ | — | done |
| sample_batched (vectorised over x) | ✅ | ✅ | — | done |
| Rejection sampling (against prior) | ✅ | ✅ | — | done |
| reject_outside_prior in sample() | ✅ | ✅ | — | done |
| MCMC: Metropolis-Hastings | ✅ | ✅ | — | done (MI4) |
| MCMC: NUTS (JAX) | ✅ | ✅ | — | done (MI4) |
| MCMC: slice sampling | ✅ | ❌ | 🟢 | MI4 |
| Variational inference (rKL, fKL) | ✅ | ❌ | 🟢 | future |
| Importance sampling posterior | ✅ | ❌ | 🟡 | future |

### Posterior API

| Feature | sbi | vbi.inference | Priority | Milestone |
|---------|-----|---------------|----------|-----------|
| sample((n,), x=x_obs) | ✅ | ✅ | — | done |
| log_prob(theta, x=x_obs) | ✅ | ✅ | — | done |
| log_prob_batched | ✅ | ✅ | — | done |
| map(x=x_obs) | ✅ | ✅ (gradient ascent) | ⚠️ | done |
| set_default_x | ✅ | ✅ | — | done |
| leakage_correction | ✅ | ✅ | — | done |
| potential / potential_fn | ✅ | ❌ | 🟡 | MI4 |
| Posterior ensemble | ✅ | ❌ | 🟢 | future |

### Prior support

| Feature | sbi | vbi.inference | Priority | Milestone |
|---------|-----|---------------|----------|-----------|
| BoxUniform | ✅ | ✅ | — | done |
| Gaussian (diagonal) | ✅ | ✅ | — | done |
| CustomPrior | ✅ | ✅ | — | done |
| MultivariateNormal (full cov) | ✅ | ✅ | — | done |
| MultipleIndependent (mixed) | ✅ | ✅ | — | done |
| RestrictedPrior (truncated) | ✅ | ✅ | — | done |
| LogNormal, Gamma, Beta, etc. | ✅ | ✅ | — | done |
| KDE wrapper | ✅ | ❌ | 🟢 | future |

### Diagnostics

| Feature | sbi | vbi.inference | Priority | Milestone |
|---------|-----|---------------|----------|-----------|
| SBC (simulation-based calibration) | ✅ | ✅ | — | done |
| TARP (expected coverage test) | ✅ | ✅ | — | done |
| C2ST (classifier 2-sample test) | ✅ | ✅ | — | done |
| LC2ST (local C2ST) | ✅ | ❌ | 🟡 | MI-diag |
| SBC rank plot | ✅ | ✅ | — | done |
| PP plot | ✅ | ❌ | 🟡 | MI-diag |
| pairplot | ✅ | ✅ | — | done |
| conditional pairplot | ✅ | ✅ | — | done |
| plot_summary (loss curves) | ✅ | ✅ (plot_loss) | — | done |
| sensitivity_analysis | ✅ | ❌ | 🟢 | future |

### Utilities

| Feature | sbi | vbi.inference | Priority | Milestone |
|---------|-----|---------------|----------|-----------|
| simulate_for_sbi helper | ✅ | ✅ | — | done |
| process_prior (validation) | ✅ | ✅ | — | done |
| get_simulations() | ✅ | ✅ | — | done |
| save / load model | ❌ (torch state_dict) | ✅ (.npz) | — | done |

---

### Gap summary

**Blocking for real use (🔴):**
~~1. APT importance weighting (multi-round `num_atoms`) — done~~
~~2. MCMC posterior refinement (MI4) — done~~

**Important but not blocking (🟡):**
1. SNLE (likelihood-based inference)
4. Custom network callables beyond `EmbeddingNet`
5. Potential / potential_fn compatibility
6. LC2ST / PP plot diagnostics

**Lower priority (🟢 — defer):**
7. SNRE, FMPE, MNPE, MNLE
8. Variational inference posterior
9. Slice sampling, HMC
10. KDE wrapper, sensitivity analysis

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
      __init__.py        # backend registry / dispatch
      jax_/              # JAX: jit + vmap + grad (MI1)
  cde.py                 # deprecated shim → re-exports from vbi.inference
```

**Backend selector** (transparent to user):
```python
SNPE(prior, density_estimator='maf', backend='auto')
# 'auto' → jax if installed, else numpy
# 'numpy' → autograd/numpy (always available)
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
| `SNPE(prior, density_estimator='maf'\|'mdn'\|'nsf', device='cpu')` | `SNPE(prior, density_estimator='maf'\|'mdn'\|'nsf', backend='auto')` | `backend` is new; `auto` selects JAX when available |
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
| Training | Mini-batch | Adam, val split, early stopping, grad clip |
| Save/load | ✅ Complete | `.npz` estimator weights/config |
| Prior integration | ✅ Complete | Prior objects + posterior prior log-prob / truncation |
| Sequential rounds | ⚠️ Partial | Rounds stored; APT proposal weighting still missing |
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

- [x] **Create `vbi/inference/` package** — move estimator classes from
      `cde.py` into `_estimators/`; keep `vbi/cde.py` as a re-export shim
      with `DeprecationWarning`
- [x] **`BoxUniform(low, high)`** — accepts numpy/list;
      `.sample((n,))` → `(n, d)`; `.log_prob(theta)` → 0 inside, `-inf` outside
- [x] **`Gaussian(mean, std)`** — `.sample((n,))` + `.log_prob(theta)`
- [x] **`CustomPrior(sample_fn, log_prob_fn)`** — user-supplied callables
- [x] **`Posterior` class** — wraps the trained estimator:
      ```python
      posterior.sample((1000,), x=x_obs)    # tuple shape, like sbi
      posterior.log_prob(theta, x=x_obs)    # params FIRST, x as kwarg
      posterior.map(x=x_obs)                # gradient ascent on log_prob
      posterior.set_default_x(x_obs)        # store default observation
      ```
      Note: `log_prob(theta, x)` internally calls `estimator.log_prob(x, theta)`
      — the argument swap is handled here so the low-level API is unchanged.
- [x] **`SNPE(prior, density_estimator='maf'|'mdn'|'nsf', backend='numpy'|'jax'|'auto')`**:
      - `append_simulations(theta, x, proposal=None)` — stores data per round
      - `train(training_batch_size=256, learning_rate=5e-4,
               validation_fraction=0.1, stop_after_epochs=20,
               clip_max_norm=5.0, max_num_epochs=2000, **kwargs)`
        maps sbi kwarg names → internal estimator kwarg names
      - `build_posterior(density_estimator=None, sample_with='direct')` →
        `Posterior`
      - `num_atoms` accepted but ignored (implement in MI3)
- [x] **`SNLE`** — stub that raises `NotImplementedError` with message
      "SNLE coming in a future release; use SNPE for now"
- [x] **Mini-batch support** in `MAFEstimator.train()` (`batch_size` kwarg,
      shuffle indices each epoch) — needed for `training_batch_size` to work
- [x] **Drop-in migration test**:
      ```python
      # Assert this works with vbi.inference.SNPE
      inference = SNPE(prior=prior, density_estimator='maf')
      inference.append_simulations(theta_np, x_np)
      est = inference.train(training_batch_size=128, stop_after_epochs=10)
      post = inference.build_posterior(est)
      s = post.sample((500,), x=x_obs_np)
      assert s.shape == (500, n_params)
      ```
- [x] `vbi/cde.py` becomes:
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

- [x] **Mini-batch training**: add `batch_size` parameter to `train()`;
      shuffle indices each epoch; default `batch_size=256`
- [x] **Save / load**: `estimator.save(path)` → pickle or npz;
      `MDNEstimator.load(path)` → class method restoring weights + config
- [x] **Logging cleanup**: replace `print(...)` with `logging.getLogger(...)`;
      verbose level controlled by `verbose: bool = False`
- [x] **Numerical stability**:
      - `logsumexp` already used in MDN; check MAF log_det underflow
      - Clip log_sigma more aggressively in MAFEstimator when needed
- [x] **API consistency**: `train()` signature unified across MDN/MAF
      (currently MAF has extra params not in base); lift common params to ABC
- [x] **Test coverage**: `test_cde.py` covers basic MDN; add
      - MAF round-trip: `log_prob` + `sample` consistency test
      - Save/load: weights survive serialization
      - Mini-batch: loss converges same as full-batch (loose tolerance)

**Effort:** Small, 1–2 days.

---

## MI0-collapse — Training stability: posterior collapse prevention

**Problem observed:** MAF training with too many epochs produces `std ≈ 0.04`
when the true posterior std is `0.30`.  The flow sharpens the distribution
indefinitely because the NLL loss keeps decreasing even as the posterior
becomes unphysically narrow.  Standard early-stopping (plateau detection)
does not catch this because the validation NLL never plateaus — it just keeps
going down as the distribution collapses.

**How sbi handles it (PyTorch):**
1. **`log_scale` clamping** — the MAF/NSF scale is clamped: `log_scale > -7`
   so `std > exp(-7) ≈ 0.001`.  Prevents total collapse but not gradual over-sharpening.
2. **Posterior std monitoring** — during training, `sbi` evaluates `posterior.sample`
   on a fixed held-out observation every N epochs and stops if the empirical std
   falls below a fraction of the prior std.
3. **Learning-rate decay** — cosine or step LR schedule means late-stage updates
   are small, limiting the amount of over-sharpening.
4. **`max_num_epochs` is intentionally set conservatively** — sbi sets
   `max_num_epochs=2**31-1` (infinite) but relies on early stopping;
   in practice, 50–200 epochs is typical.  The user is not expected to tune this.

**Tasks:**

- [x] **LR schedule**: add `lr_schedule='cosine'|'step'|None` to `MAFEstimator.train()`;
      cosine annealing from `learning_rate` to `learning_rate * 0.01` over training
- [x] **`log_scale` clamp**: in `_made_forward`, `log_sig = clip(out, -7, 7)` already
      exists — verify it is effective and add a `min_log_scale` constructor kwarg
- [x] **Posterior std monitoring**: add `monitor_collapse=True` to `train()`;
      every `check_every=10` epochs, draw `n_check=200` samples at a fixed
      `x_check` observation; if `samples.std() < prior_std * collapse_threshold`
      (default 0.05), restore best weights and stop
- [x] **Auto `max_num_epochs`**: when `monitor_collapse=True`, the user does not
      need to set `max_num_epochs` — the collapse monitor + plateau early-stopping
      handle termination automatically
- [x] **`SNPE.train()` kwarg**: expose `monitor_collapse=True` and `x_check=None`
      (auto-selected from training data if not provided)
- [x] **Tests**: verify that with `monitor_collapse=True`, std error on the
      1-D Gaussian demo stays < 0.15 regardless of `max_num_epochs`

**Effort:** Small, 1-2 days.

---

## MI0-NSF — Neural Spline Flow density estimator

**Goal:** Add NSF, which is sbi's best-performing density estimator.
Rational-quadratic spline transforms with pure NumPy/autograd.

**Tasks:**
- [x] `vbi/inference/_estimators/nsf.py` — RQ-spline MADE block
- [x] `NSFEstimator` class with same API as `MAFEstimator`
- [x] Register `density_estimator='nsf'` in `SNPE`
- [x] Tests: NSF log_prob finite and loss lower than MAF on moons dataset

**Effort:** Medium, 3-4 days.

---

## MI0-embed — Embedding / summary network

**Goal:** Allow a learned neural network to compress high-dimensional raw x
(e.g. time series) into a low-dimensional summary before conditioning.
This is essential for using raw simulator output without hand-crafted features.

```python
SNPE(prior, density_estimator='nsf', embedding_net=MLP(x_dim, 20))
```

**Tasks:**
- [x] `EmbeddingNet` class: MLP that maps `(n, x_dim)` → `(n, embed_dim)`
- [x] `SNPE(..., embedding_net=None)` kwarg; applied inside `train()` to features
- [x] Default identity (no embedding) leaves behaviour unchanged
- [x] Tests: embedding reduces feature dim correctly; loss converges

**Effort:** Small, 1 day.

---

## MI0-rejection — Rejection sampling posterior + prior truncation

**Goal:** Samples that fall outside the prior support are rejected.
Adds `reject_outside_prior=True` to `Posterior.sample()` and `leakage_correction`.

**Tasks:**
- [x] `Posterior.sample(..., reject_outside_prior=True)` — oversample then filter
- [x] `Posterior.leakage_correction(x_obs)` → scalar leakage estimate
- [x] `build_posterior(..., sample_with='rejection')` — explicit rejection path
- [x] Tests: samples from BoxUniform prior always inside bounds

**Effort:** Small, 1 day.

---

## MI0-priors — Extended prior distributions

**Goal:** Match the prior types users expect from scipy/sbi.

**Tasks:**
- [x] `MultivariateNormal(mean, cov)` — full covariance Gaussian
- [x] `LogNormal(mean, std)` — log-normal marginals
- [x] `Gamma(concentration, rate)` — positive-valued parameters
- [x] `Beta(alpha, beta)` — [0,1]-valued parameters
- [x] `MultipleIndependent([prior1, prior2, ...])` — product of independent priors
      (mix different types per parameter dimension)
- [x] `RestrictedPrior(base_prior, constraint_fn)` — truncated prior
- [x] All priors: `.sample((n,))`, `.log_prob(theta)`, `.dim` property
- [x] Tests: each prior samples correct shape and log_prob is finite inside support

**Effort:** Small-medium, 2 days.

---

## MI0-utils — Simulation and training utilities

**Goal:** QoL helpers that sbi users expect.

**Tasks:**
- [x] `simulate_for_sbi(simulator_fn, prior, num_simulations, seed=None)` →
      `(theta, x)` — runs simulator in a loop, handles errors gracefully
- [x] `SNPE.get_simulations(starting_round=0)` → `(theta, x, proposal)` —
      retrieve stored simulation data
- [x] `SNPE.train(..., resume_training=True)` — warm start from current weights
- [x] `process_prior(prior)` — validate that prior has `.sample` and `.log_prob`
- [x] Tests: simulate_for_sbi returns correct shapes

**Effort:** Small, 1 day.

---

## MI-diag — Diagnostics

**Goal:** Statistical tools to validate that the trained posterior is correct.
These are the most critical tools for scientific use.

**Tasks:**

### SBC (Simulation-Based Calibration)
- [x] `run_sbc(posterior, simulator, prior, num_sbc_runs, num_posterior_samples)`
      → `{ranks: array, dap_samples: array}`
- [x] `check_sbc(ranks)` → `{uniformity_pvalue, c2st_ranks}`
- [x] `sbc_rank_plot(ranks)` — per-parameter rank histograms

### Coverage (TARP)
- [x] `run_tarp(posterior, simulator, prior, num_runs)`
      → `{alphas, ecp}` (expected coverage probability)
- [x] `check_tarp(alphas, ecp)` → coverage summary
- [x] `plot_tarp(alphas, ecp)` — coverage plot

### Classifier 2-Sample Test (C2ST)
- [x] `c2st(samples_p, samples_q, seed=None)` → accuracy ∈ [0.5, 1.0]
      (0.5 = indistinguishable = good posterior)
- [x] Requires sklearn LogisticRegression (already available)

### Posterior visualisation
- [x] `pairplot(samples, points=None, limits=None, labels=None)` →
      matplotlib Figure — triangle plot of marginals and pairwise joints
- [x] `conditional_pairplot(posterior, x_obs, prior)` — condition on observation
- [x] `plot_loss(loss_history, val_loss_history)` — training curves

**Effort:** Medium, 4-5 days.  SBC and pairplot are the most impactful.

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

- [x] Create `vbi/cde_jax.py` (or add `backend='jax'` dispatch in `cde.py`)
- [x] Functional Adam optimizer (pure functions, no mutation):
      ```python
      def adam_update(weights, grads, m, v, t, lr): ...
      ```
- [x] JIT-compile training loop: `jax.jit(grad(loss_fn))`
- [x] `jax.random` PRNG threading through sample/train
- [x] `vmap`-compatible `log_prob` and `sample` for batch conditions
- [x] `jax.grad` through the full `log_prob` (for gradient-based posterior
      exploration)
- [x] Mini-batch training with `jnp.take`
- [x] Data normalization inside JIT-safe transforms
- [x] Benchmark: initial autograd vs JAX CPU demo (`08_backend_benchmark.py`);
      large-N / GPU benchmarks still pending
- [x] Tests: numerical match autograd vs JAX (rtol=1e-4)
- [x] Fallback: if JAX not installed, silently use autograd

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

## MI6 — End-to-end `VBIInference` API

**Goal:** One object owns the full workflow: prior sampling → sweep simulation
→ feature extraction → SNPE training → posterior.  No manual glue code.
Expensive simulations can be cached to disk so features can be recomputed
without re-running the sweeper.

### Resolved design decisions

**Q1 — Sweeper return format with pipeline:**
`NumpySweeper.run(dur)` returns `(labels, values)` where
`values.shape == (n_samples, n_params + n_features)`, params first.
`simulate_for_vbi_sweep` therefore calls `sweeper.run()` and slices:
`theta = values[:, :n_params]`, `x = values[:, n_params:]`.

**Q2 — `SimulationSpec.from_dict`:** Does not exist.
Step 4 must add a minimal `SimulationSpec.from_dict(d)` to
`vbi/simulator/spec/simulation.py`.

**Q3 — `param_names` location:** Add `param_names: list[str] | None = None`
to all prior constructors in `_prior.py` (see Step 0).
Rationale: the prior semantically owns its parameter names; this keeps
`VBIInference` clean, makes pairplots automatic, and matches how sbi
users think about priors.

---

### Target API

```python
from vbi.inference import VBIInference, BoxUniform
import numpy as np

prior = BoxUniform(
    low=np.array([0.5, -5.5]),
    high=np.array([5.0, -3.0]),
    param_names=["G", "eta"],       # NEW — see Step 0
)

inf = VBIInference(
    sim_spec          = sim_spec,
    prior             = prior,
    pipeline          = feature_pipeline,
    density_estimator = "maf",
    sim_backend       = "numba",
    backend           = "auto",
)

# --- Round 1: simulate → cache raw → extract features → train ---
theta, x = inf.simulate(
    num_simulations = 2000,
    duration        = 5000.0,
    seed            = 0,
    cache_dir       = "./sim_cache/r1/",   # optional; omit to skip caching
    chunk_size      = 500,                 # simulations per chunk (GPU memory control)
)
estimator = inf.train(training_batch_size=256, stop_after_epochs=30)
posterior  = inf.build_posterior(estimator)

# --- Re-extract with different features, no re-simulation ---
new_pipeline = FeaturePipeline(new_cfg, signal="tavg", t_cut=1000.0)
theta2, x2 = VBIInference.extract_from_cache("./sim_cache/r1/", new_pipeline)

# --- Round 2: simulate from posterior ---
theta_r2, x_r2 = inf.simulate(
    num_simulations = 500,
    duration        = 5000.0,
    proposal        = posterior,
    x_obs           = x_obs,
    cache_dir       = "./sim_cache/r2/",
)
estimator2 = inf.train()
posterior2  = inf.build_posterior(estimator2)

# --- Posterior use ---
samples = posterior2.sample((2000,), x=x_obs)
inf.save("run_mpr_g.npz")

# --- Reload inference state (not raw cache) ---
inf2 = VBIInference.load("run_mpr_g.npz", sim_spec=sim_spec, pipeline=feature_pipeline)
```

### File layout

```
vbi/inference/
  _vbi_inference.py    NEW — VBIInference class
  _utils.py            ADD simulate_for_vbi_sweep(), extract_from_cache()
  __init__.py          ADD VBIInference export
vbi/simulator/spec/
  simulation.py        ADD SimulationSpec.from_dict()
```

---

### Step 0 — Add `param_names` to prior classes  *(prerequisite)*

**File:** `vbi/inference/_prior.py`

All prior classes gain an optional `param_names` argument.  When `None`,
auto-generate `["p0", "p1", ...]` at call time based on the sample dimension.

```python
class BoxUniform:
    def __init__(self, low, high, param_names=None):
        ...
        self.param_names = param_names  # list[str] | None

    @property
    def _resolved_param_names(self) -> list[str]:
        if self.param_names is not None:
            return list(self.param_names)
        d = np.asarray(self.low).shape[0]
        return [f"p{i}" for i in range(d)]
```

**Tasks:**

- [ ] Add `param_names: list[str] | None = None` to `BoxUniform.__init__`
- [ ] Same for `Gaussian`, `MultivariateNormal`, `MultipleIndependent`,
      `LogNormal`, `Gamma`, `Beta`, `RestrictedPrior`
- [ ] Add `_resolved_param_names` property to each (or a shared mixin)
- [ ] No change to `sample()` or `log_prob()` — purely additive

**Effort:** ~1 hour. Purely additive, no logic changes.

---

### Step 1 — Simulation bridge in `_utils.py`

**New function:** `simulate_for_vbi_sweep`

Bridges `prior.sample(n)` → `SweepSpec(params=theta, pipeline=pipeline)`
→ `Sweeper.run(duration)` → slice `(labels, values_concat)` → `(theta, x)`.

```python
def simulate_for_vbi_sweep(
    sim_spec,
    prior,
    pipeline,
    num_simulations: int,
    duration: float,
    sim_backend: str = "numba",
    seed: int | None = None,
    proposal=None,     # None → sample from prior; Posterior → sample from it
    x_obs=None,        # required when proposal is not None
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Returns (theta, x, param_names, feature_labels).
    theta : (n, d_theta) float64
    x     : (n, d_x)     float64
    """
```

**Tasks:**

- [ ] Sample `theta` from `prior._resolved_param_names` / `prior.sample(n, seed)`
      (or `proposal.sample` when proposal is provided)
- [ ] Build `SweepSpec(params=theta, param_names=param_names, pipeline=pipeline)`
- [ ] Instantiate `Sweeper(sim_spec, sweep_spec, backend=sim_backend)`
- [ ] Call `labels, values = sweeper.run(duration)`
  - [ ] `values.shape == (n_samples, n_params + n_features)` — confirmed from
        `numpy_/sweeper.py` (params first, features after)
- [ ] Slice: `theta_out = values[:, :n_params]`, `x_out = values[:, n_params:]`
  - [ ] `feature_labels = labels[n_params:]`
- [ ] Drop NaN rows (any non-finite value in `x_out`); log how many were dropped
- [ ] Return `theta_out, x_out, param_names, feature_labels` (all float64)
- [ ] Export from `vbi/inference/__init__.py`
- [ ] Test: `tests/test_simulate_for_vbi_sweep.py`
  - [ ] Numba backend, MPR model, 20 simulations, FC features → shapes correct
  - [ ] `feature_labels` is a non-empty list of strings
  - [ ] Same seed → identical output

**Effort:** ~half day.

---

### Step 2 — Simulation cache: store raw recordings, re-extract features

Time-consuming sweeps (especially CUDA/C++) should be cached so the user can
change feature parameters without re-running the simulator.

**Cache format:** a directory of chunked `.npz` files + a `metadata.json`.

```
sim_cache/r1/
  metadata.json          n_samples, chunk_size, param_names, duration,
                         sim_backend, dt, n_nodes, n_sv, monitor_period
  chunk_000.npz          theta (chunk,d_θ), ts (chunk,n_record,n_sv,n_nodes), t (n_record,)
  chunk_001.npz
  ...
```

Memory contract:
- Each chunk is independent — only one chunk needs to be in RAM at a time.
- `chunk_size` controls GPU memory pressure: for CUDA with 80 nodes × 5 000 steps,
  `chunk_size=200` uses ~640 MB float32; `chunk_size=500` uses ~1.6 GB.
- Feature extraction at load time processes one chunk, runs `pipeline.extract()`,
  discards the raw `ts`, accumulates only the feature vector.

```python
# Run sweep + write cache
theta, x = inf.simulate(
    num_simulations=5000,
    duration=5000.0,
    cache_dir="./sim_cache/r1/",
    chunk_size=500,
)

# Later: re-extract with different pipeline, no simulation
theta2, x2 = VBIInference.extract_from_cache("./sim_cache/r1/", new_pipeline)
```

**New function:** `simulate_for_vbi_sweep_cached` (internal, called by `simulate`
when `cache_dir` is set):

```python
def simulate_for_vbi_sweep_cached(
    sim_spec, prior, pipeline, num_simulations, duration,
    cache_dir, chunk_size=500, sim_backend="numba", seed=None, proposal=None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
```

**New static method:** `VBIInference.extract_from_cache(cache_dir, pipeline)`

**Tasks:**

- [ ] `simulate_for_vbi_sweep_cached` in `_utils.py`:
  - [ ] Create `cache_dir` if it doesn't exist
  - [ ] Sample all `theta` upfront (or per-chunk if proposal needs iterative sampling)
  - [ ] Loop over `ceil(n / chunk_size)` chunks:
    - [ ] Build `SweepSpec(params=theta_chunk, pipeline=None)` — raw output, no pipeline
    - [ ] Run `sweeper.run_raw(duration)` → `(ts, t)` arrays
          (need to verify/add a `run_raw` path that returns raw time series not features)
    - [ ] Write `chunk_{i:03d}.npz` with `theta`, `ts`, `t`
    - [ ] Free `ts` from memory before next chunk
  - [ ] Write `metadata.json`
  - [ ] Extract features from each chunk (in a second pass or inline) to return `(theta, x)`
- [ ] `VBIInference.extract_from_cache(cache_dir, pipeline) -> (theta, x)` (static method):
  - [ ] Read `metadata.json`
  - [ ] Loop chunks: load `theta`, `ts`, `t`; call `pipeline.extract({"tavg": (t, ts)})`
        per sample; accumulate `(theta_rows, x_rows)`
  - [ ] Return stacked `(theta, x)` arrays
- [ ] Verify that `NumpySweeper` and `NumbaSweeperCPU` have a path to return raw
      `(ts, t)` without pipeline — `_run_raw` exists in Numba sweeper; check numpy
  - [ ] If missing from numpy sweeper: add `run_raw(duration)` that skips pipeline
        and returns `(param_sets, raw_ts, t_array)`
- [ ] Tests: `tests/test_sim_cache.py`
  - [ ] `simulate(..., cache_dir=tmp_path)` writes expected files
  - [ ] `extract_from_cache(tmp_path, pipeline)` returns same `x` as direct sweep
  - [ ] `extract_from_cache(tmp_path, new_pipeline)` returns different `x` (different features)
  - [ ] Two-chunk sweep: both chunks loaded and concatenated correctly

**Effort:** ~1.5 days. The main work is the chunked write loop and verifying/adding
`run_raw` to the sweepers.

---

### Step 3 — `VBIInference` core: `__init__`, `simulate`, `train`, `build_posterior`

**New file:** `vbi/inference/_vbi_inference.py`

```python
class VBIInference:
    def __init__(
        self,
        sim_spec,
        prior,
        pipeline,
        density_estimator: str = "maf",
        sim_backend: str = "numba",
        backend: str = "auto",
        show_progress_bars: bool = True,
        embedding_net=None,
    ):
        self._sim_spec    = sim_spec
        self._prior       = prior
        self._pipeline    = pipeline
        self._sim_backend = sim_backend
        self._snpe = SNPE(
            prior=prior,
            density_estimator=density_estimator,
            backend=backend,
            show_progress_bars=show_progress_bars,
            embedding_net=embedding_net,
        )
        self._feature_labels: list[str] | None = None
        self._param_names: list[str] | None = None
        self._default_train_kwargs: dict = {}
```

**Tasks:**

- [ ] `simulate(num_simulations, duration, seed=None, proposal=None, x_obs=None,
               cache_dir=None, chunk_size=500)`
  - [ ] If `cache_dir` is None: call `simulate_for_vbi_sweep(...)` (Step 1)
  - [ ] If `cache_dir` is set: call `simulate_for_vbi_sweep_cached(...)` (Step 2)
  - [ ] Store `_feature_labels` and `_param_names` from first call
  - [ ] Call `self._snpe.append_simulations(theta, x)`
  - [ ] Return `(theta, x)`
- [ ] `train(**train_kwargs) -> ConditionalDensityEstimator`
  - [ ] Merge `self._default_train_kwargs` with explicit `train_kwargs`
        (explicit overrides defaults)
  - [ ] Delegate: `return self._snpe.train(**merged_kwargs)`
  - [ ] Raise `RuntimeError` if `simulate()` was never called
- [ ] `build_posterior(estimator=None, sample_with="direct", **kwargs) -> Posterior`
  - [ ] Delegate: `return self._snpe.build_posterior(estimator, sample_with, **kwargs)`
- [ ] `get_simulations() -> tuple[np.ndarray, np.ndarray]`
  - [ ] Delegate to `self._snpe.get_simulations()`
- [ ] `staticmethod extract_from_cache(cache_dir, pipeline)` — proxy to `_utils`
- [ ] Add to `vbi/inference/__init__.py` and `__all__`
- [ ] Tests: `tests/test_vbi_inference.py`
  - [ ] End-to-end (no cache): `simulate(50)` → `train(stop_after_epochs=5)`
        → `build_posterior()` → `posterior.sample((100,), x=x_obs)` → shape correct
  - [ ] With cache: same round-trip, verify `cache_dir` files exist

**Effort:** ~1 day.

---

### Step 4 — Save / load checkpointing

Persist inference state (not raw cache) so training can be resumed.

**What to save:** estimator weights, `(theta, x)` rounds, `feature_labels`,
`param_names`, constructor kwargs (`density_estimator`, `sim_backend`, `backend`).

**What NOT to save:** `sim_spec`, `pipeline` — Python objects; user re-supplies them.

```python
inf.save("checkpoint.npz")
inf2 = VBIInference.load("checkpoint.npz", sim_spec=sim_spec, pipeline=feature_pipeline)
```

**Tasks:**

- [ ] `save(path) -> None`
  - [ ] Write estimator weights to a temporary `.npz`, embed as bytes array
  - [ ] Pack into a single `.npz`: `theta_all`, `x_all`, `feature_labels`,
        `param_names`, `density_estimator`, `sim_backend`, `backend`, `n_rounds`
- [ ] `classmethod load(path, sim_spec, pipeline, prior=None) -> VBIInference`
  - [ ] Load `.npz`, reconstruct `VBIInference.__init__` from saved kwargs
  - [ ] Restore estimator via `ConditionalDensityEstimator.load`
  - [ ] Re-inject `(theta, x)` via `_snpe.append_simulations`
  - [ ] Restore `_feature_labels` and `_param_names`
- [ ] Tests:
  - [ ] Save → load → `get_simulations()` identical
  - [ ] Save (trained) → load → `build_posterior()` → samples same as pre-save (same seed)

**Effort:** ~half day.

---

### Step 5 — Config loading: `from_config`

```yaml
# vbi_config.yaml
sim:
  model: MPR
  connectivity: data/SC_68.npz
  node_params: {eta: -4.6, tau: 1.0}
  dt: 0.1
  monitors: [tavg]
  monitor_period: 1.0

prior:
  type: BoxUniform
  low:  [0.5, -5.5]
  high: [5.0, -3.0]
  param_names: [G, eta]     # passed to BoxUniform constructor (Step 0)

pipeline:
  features: [calc_fc, calc_fcd]
  signal: tavg
  t_cut: 500.0

inference:
  density_estimator: maf
  sim_backend: numba
  backend: auto
  training:
    training_batch_size: 256
    stop_after_epochs: 30
    learning_rate: 5.0e-4
```

**Tasks:**

- [ ] `SimulationSpec.from_dict(d: dict) -> SimulationSpec` in
      `vbi/simulator/spec/simulation.py` (does not exist — must add)
  - [ ] Parse `model`, `connectivity`, `node_params`, `dt`, `monitors`,
        `monitor_period`; use existing `SimulationSpec.__init__` internally
- [ ] `classmethod VBIInference.from_config(config: str | dict) -> VBIInference`
  - [ ] Load YAML string → dict (via `yaml`, fallback to `json`)
  - [ ] `sim` block → `SimulationSpec.from_dict(...)`
  - [ ] `prior` block → dispatch on `type` to existing prior classes in `_prior.py`
  - [ ] `pipeline` block → `get_features_by_given_names` + `FeaturePipeline(...)`
  - [ ] `inference` block → `VBIInference.__init__` kwargs
  - [ ] Store `inference.training` as `self._default_train_kwargs`
- [ ] Tests: `tests/test_vbi_inference_config.py`
  - [ ] Minimal YAML → `from_config` → `simulate(20)` → no crash
  - [ ] `prior.param_names` populated from config
  - [ ] Training kwargs from config pass through to `train()`

**Effort:** ~1 day. Bulk is `SimulationSpec.from_dict`.

---

### Step 6 — Diagnostic helpers

Thin forwarding so user doesn't need to import `_diagnostics` directly.

**Tasks:**

- [ ] `plot_loss() -> Figure`
  - [ ] Retrieve loss history from estimator (check actual attribute names
        in `ConditionalDensityEstimator` before implementing)
  - [ ] Delegate to `plot_loss(train_losses, val_losses)` in `_diagnostics`
- [ ] `pairplot(x_obs, num_samples=1000, **kwargs) -> Figure`
  - [ ] Sample from trained posterior at `x_obs`
  - [ ] Delegate to `pairplot(samples, labels=self._param_names, **kwargs)`
- [ ] `run_sbc(num_sbc_runs=500, num_posterior_samples=100) -> dict`
  - [ ] Build `simulator_fn` from `sim_spec + pipeline` internally
  - [ ] Delegate to `run_sbc(posterior, simulator_fn, prior, ...)` in `_diagnostics`
- [ ] Tests: `plot_loss()` and `pairplot()` return matplotlib Figures without error

**Effort:** ~half day.

---

### Step 7 — Tests and example notebook

**Tests** (`tests/test_vbi_inference.py`):

- [ ] Full round-trip: `simulate(50)` → `train(5 epochs)` → `build_posterior()`
      → `sample((100,), x=x_obs)` — shape assertions only, MPR + numba
- [ ] Cache round-trip: same as above with `cache_dir`; verify
      `extract_from_cache` with alternate pipeline gives different `x`
- [ ] Sequential rounds: two `simulate + train` calls; `get_simulations()` doubles
- [ ] `save` / `load` preserves training state
- [ ] `from_config` minimal YAML
- [ ] `plot_loss()` returns Figure

**Example notebook** (`examples/vbi_inference_mpr.ipynb`):

- [ ] MPR model, SC_68, `BoxUniform(param_names=["G","eta"])`
- [ ] 1-round SNPE, 500 simulations (numba), FC features, MAF, pairplot
- [ ] Cache demo: re-extract with FCD features, no re-simulation
- [ ] 2-round SNPE: posterior-focused round 2, TARP coverage check
- [ ] `save` / `load` demo

---

### Implementation order

```
Step 0  param_names on priors          (~1 hour)    prerequisite for Steps 1+
Step 1  simulate_for_vbi_sweep         (~0.5 day)   core bridge, unblocks Step 3
Step 2  simulation cache               (~1.5 days)  run_raw + chunked write + extract
Step 3  VBIInference core              (~1 day)     first working end-to-end
Step 4  save / load                    (~0.5 day)   needed for real use
Step 5  from_config                    (~1 day)     reproducibility
Step 6  diagnostic helpers             (~0.5 day)   polish
Step 7  tests + notebook               (~1 day)     validation
```

**Total effort: ~6 days of focused work.**

---

**Effort:** ~6 days total.

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
| ~~MI-numba~~ | ~~Numba JIT~~ | ~~skipped — dead end without autograd~~ | — |
| **MI1** | JAX | GPU, vmap, jit, gradient through posterior | `jax` |
| **MI2** | all | prior integration, Posterior object (part of MI-API) | MI-API |
| **MI3** | all | sequential rounds, multi-round SNPE | MI-API + MI2 |
| **MI4** | numpy + JAX | MCMC: MH (numpy), NUTS (JAX) | MI1 + MI2 |
| **MI6** | all | VBIInference end-to-end integration | MI1 + MI2 + MI3 |
| **MI5** | torch (optional) | large models, PyTorch ecosystem | MI-API |

## Priority order

**Current focus: finish sequential / MCMC parity and VBI integration.**

```
★ DONE:  MI-API        sbi-compatible interface, MAF, MDN, BoxUniform, Gaussian,
                        Posterior (sample/log_prob/map), mini-batch, save/load
         MI0-collapse  cosine LR, log_scale clamp, posterior-collapse monitor
         MI0-rejection reject_outside_prior, leakage_correction, sample_with='rejection'
         MI0-embed     EmbeddingNet (jointly trained MLP summary network)
         MI0-utils     simulate_for_vbi, get_simulations, resume_training, process_prior
          MI0-priors   MultivariateNormal, MultipleIndependent, Beta, Gamma, RestrictedPrior
          MI-diag      SBC, TARP, C2ST, pairplot, plot_loss
          MI0-NSF      NSF density estimator (numpy/autograd + JAX)
          MI1          JAX backend (MDN, MAF, NSF), auto backend, CPU-safe device selection

  DONE (recent):
          MI3   Sequential rounds with APT importance weights (num_atoms)
          MI4   MCMC posterior: MH (numpy) + NUTS (JAX) + R-hat/ESS diagnostics

  NEXT:
          MI6   End-to-end VBIInference API

  SKIPPED / DEFERRED:
          MI-numba  Numba JIT — no autograd, finite-diff gradients are impractical
          MI5       Torch backend (optional, post-JAX)
          SNLE, SNRE, FMPE, MNPE
```

### Dependency graph

```
MI-API + MI0-*
  │
  ├──→ MI1 (JAX) ──→ MI4 (MCMC/NUTS)
  │                        │
  ├──→ MI3 (sequential) ───┤
  │                        │
  └──→ MI6 (end-to-end) ───┘

  MI5 (torch, optional, post-JAX)
```

---

## Comparison with `sbi`

| Feature | `sbi` | `vbi.inference` |
|---------|-------|----------------|
| MDN | ✅ | ✅ done |
| MAF | ✅ | ✅ done |
| Embedding / summary network | ✅ | ✅ done (EmbeddingNet) |
| Mini-batch training | ✅ | ✅ done |
| Save / load | ✅ | ✅ done |
| Rejection sampling posterior | ✅ | ✅ done |
| simulate_for_sbi helper | ✅ | ✅ done (simulate_for_vbi) |
| get_simulations | ✅ | ✅ done |
| resume_training (warm start) | ✅ | ✅ done |
| BoxUniform, Gaussian, CustomPrior | ✅ | ✅ done |
| MultivariateNormal, Beta, Gamma | ✅ | ✅ done |
| MultipleIndependent, RestrictedPrior | ✅ | ✅ done |
| NSF (neural spline flow) | ✅ | ✅ done |
| SBC, TARP, C2ST diagnostics | ✅ | ✅ done |
| pairplot, plot_loss | ✅ | ✅ done |
| Sequential rounds (SNPE-C / APT) | ✅ | ✅ done (MI3) |
| MCMC refinement (MH + NUTS) | ✅ | ✅ done (MI4) |
| GPU training | ✅ (torch.cuda) | ⚠️ JAX backend exists; GPU needs environment validation |
| vmap batch eval | ✅ | ✅ batched posterior APIs + JAX backend |
| Gradient through posterior | Limited | ✅ JAX estimator path |
| Dependency size | ~2 GB | ~50 MB (numpy+autograd) |
| Integration with VBI sim | ❌ manual | ❌ MI6 |
| Numpy speed (small N) | Slower | ✅ already faster |

---

## Open design questions

1. **Autograd vs JAX as default**: After MI1, should JAX replace autograd as
   the default backend? Proposal: keep `backend='autograd'` as always-available
   fallback; JAX becomes default when installed (`backend='auto'`).

2. ~~**NSF in autograd vs JAX**~~: resolved — NSF exists in both the
   numpy/autograd and JAX estimator maps.

3. **`MAFEstimator0` deprecation**: The original simpler MAF should be
   deprecated at the start of MI1 and removed one release cycle after.

4. ~~**Embedding network**~~: resolved — EmbeddingNet implemented in MI0-embed.

5. **Amortized vs sequential**: the current CDE is fully amortized
   (one network for all conditions). Sequential SBI (MI3) breaks amortization.
   Decision: keep both paths; amortized remains the default.
