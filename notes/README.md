# VBI Transition Note: Unified Simulator and Inference Stack

This note is a draft README for the ongoing transition toward the new `vbi.simulator` and `vbi.inference` APIs. It is kept in `notes/` for now so it can be reviewed and refined before replacing or restructuring the root `README.md`.

## Recommended Placement

For now, keep this file as:

```text
notes/README.md
```

When the transition is ready for users, promote a shorter version into the root `README.md` and move the longer developer/migration details into documentation pages, for example:

```text
README.md                         # user-facing overview and quick start
docs/simulator/index.md           # simulator API guide
docs/inference/index.md           # inference API guide
docs/migration/simulator.md       # migration from older model-specific paths
notes/                            # internal reviews, milestones, design notes
```

The root `README.md` should eventually introduce `vbi.simulator` and `vbi.inference` as the preferred public APIs, while still documenting that older model-specific paths remain available during the transition.

## Overview

VBI is moving toward a unified simulation and inference architecture. The goal is to make virtual brain model inversion easier to maintain, easier to validate across computational backends, and easier to use in simulation-based inference workflows.

The transition has two main parts:

- `vbi.simulator`: a backend-agnostic simulator API built around explicit model, integration, coupling, monitor, sweep, and stimulation specifications.
- `vbi.inference`: a torch-free, NumPy/JAX-oriented SBI API with familiar `sbi`-style workflows for training neural density estimators and sampling posteriors.

Older model-specific modules should not be removed immediately. They remain useful for compatibility, examples, and validation. The new stack should replace them gradually as each model and workflow reaches parity.

## Why This Transition Exists

The previous simulator organization grew around individual model implementations. That made it possible to move quickly, but it also created repeated logic across models and backends:

- integration loops were duplicated;
- monitor behavior could drift between implementations;
- parameter sweeps were not consistently represented;
- backend-specific behavior was exposed too early to users;
- inference workflows had to bridge several simulator conventions.

The new architecture separates the model definition from the execution backend. A model is described once as a `ModelSpec`; each backend then implements the same `Simulator` and `Sweeper` contract.

The intended user-facing result is simple:

```python
sim = Simulator(spec, backend="numba")
result = sim.run(duration=1000.0)
```

and:

```python
sweeper = Sweeper(spec, sweep_spec, backend="cuda")
labels, values = sweeper.run(duration=5000.0)
```

Changing backend should usually mean changing one string, not rewriting the workflow.

## New Simulator API

The public simulator entry points are:

```python
from vbi.simulator import Simulator, Sweeper
```

The simulator is configured with spec objects:

```python
from vbi.simulator.spec import (
    SimulationSpec,
    IntegratorSpec,
    CouplingSpec,
    MonitorSpec,
    SweepSpec,
)
```

Models live under:

```python
from vbi.simulator.models.mpr import mpr
```

### Single Simulation

```python
import numpy as np

from vbi.simulator import Simulator
from vbi.simulator.models.mpr import mpr
from vbi.simulator.spec import (
    SimulationSpec,
    IntegratorSpec,
    CouplingSpec,
    MonitorSpec,
)

n_nodes = 4
weights = np.zeros((n_nodes, n_nodes))

spec = SimulationSpec(
    model=mpr,
    integrator=IntegratorSpec(method="heun", dt=0.1, stochastic=False),
    coupling=CouplingSpec(kind="linear", a=1.0, b=0.0),
    monitors=(MonitorSpec("raw"),),
    weights=weights,
    node_params={"eta": np.full(n_nodes, -4.6), "G": np.array([1.0])},
)

sim = Simulator(spec, backend="numba")
result = sim.run(duration=100.0)

t, x = result["raw"]
```

### Parameter Sweeps

Parameter sweeps are first-class because they are the main workload for SBI.

```python
from vbi.simulator import Sweeper
from vbi.simulator.spec import SweepSpec

sweep_spec = SweepSpec(
    params={"eta": np.linspace(-6.0, -3.0, 32)},
    t_cut=500.0,
)

sweeper = Sweeper(spec, sweep_spec, backend="numba")
results = sweeper.run(duration=2000.0)
```

With a feature pipeline, `Sweeper.run()` can return tabular training data:

```python
labels, values = sweeper.run(duration=5000.0)
```

where `values` contains the swept parameters and extracted features.

## Backend Strategy

The simulator is designed around a shared API with multiple execution backends.

| Backend | Intended role |
| --- | --- |
| `numpy` | Reference implementation and easiest debugging path |
| `numba` | CPU JIT backend for routine sweeps |
| `cpp` | Compiled CPU backend for high-throughput production sweeps |
| `jax` | Differentiable backend, vectorized sweeps, gradient-based workflows |
| `cuda` | GPU sweep backend for large parallel parameter grids |

The migration rule should be:

1. Validate behavior against `numpy`.
2. Use `numba` or `cpp` for faster CPU workloads.
3. Use `jax` when differentiability or JAX vectorization matters.
4. Use `cuda` for large sweep batches when CUDA validation covers the required monitor/pipeline path.

Backend limitations should be explicit. If a backend does not support a model, coupling, monitor, or sweep parameter, it should raise a clear error rather than silently producing different semantics.

## Benchmarking Snapshot

The transition documentation should include a small number of clear benchmark figures. The goal is not to show every timing experiment, but to make the backend tradeoffs intuitive for users choosing between `numpy`, `numba`, `cpp`, `jax`, and `cuda`.

Recommended figure set:

1. Single simulation runtime across backends.
2. Parameter sweep throughput across backends.
3. Scaling with number of sweep samples.
4. Optional GPU-focused benchmark for CUDA/JAX when GPU hardware is available.

Placeholder:

```text
![Simulator backend benchmark placeholder](../docs/examples/simulator_models/outputs/backend_benchmark_placeholder.png)
```

Suggested caption:

> Runtime and throughput comparison for the unified `vbi.simulator` API. The NumPy backend is the reference implementation; Numba and C++ target fast CPU execution; JAX supports vectorized and differentiable workflows; CUDA targets large independent parameter sweeps. Exact speedups depend on model complexity, number of nodes, monitor choice, sweep size, and available hardware.

Suggested layout for the final figure:

- Panel A: single-run runtime, lower is better.
- Panel B: sweep throughput in simulations per second, higher is better.
- Panel C: scaling with number of sweep samples.
- Panel D: memory or compile overhead, if relevant.

Notes for benchmark interpretation:

- Include first-call and warmed-call timings separately for JIT/compiled backends where compilation overhead matters.
- Report hardware, Python version, backend versions, GPU model, and CUDA/JAX configuration.
- Use the same `SimulationSpec`, duration, monitor, and sweep grid across backends.
- For CUDA and JAX, avoid very small sweep sizes when reporting throughput because launch/compile overhead dominates.
- Keep benchmark figures separate from parity/validation figures. A fast backend is only useful when it preserves the same simulator contract.

## Model Migration Strategy

The old model-specific paths should be replaced gradually, not removed in one step.

Recommended migration phases:

1. Keep existing model modules and examples working.
2. Add a `vbi.simulator.models.<model>` `ModelSpec` for each model.
3. Validate `numpy` output against the previous implementation.
4. Validate accelerated backends against the `numpy` reference.
5. Update examples to prefer `vbi.simulator`.
6. Mark older paths as compatibility paths after equivalent simulator workflows exist.
7. Deprecate only after at least one stable release cycle with clear migration examples.

This avoids breaking current users while making the new simulator stack the default path for new work.

## Current Simulator Development Highlights

The new simulator stack introduces:

- explicit `SimulationSpec` objects for model, integrator, coupling, monitors, connectivity, delays, node parameters, and stimulation;
- a shared `Simulator` API for single runs;
- a shared `Sweeper` API for parameter sweeps;
- `ModelSpec` definitions for multiple neural mass and oscillator models;
- monitor support for raw, subsample, temporal average, global average, and BOLD-style output;
- support for deterministic and stochastic Euler/Heun integration where implemented by the backend;
- backend dispatch for NumPy, Numba, C++, JAX, and Numba-CUDA;
- cross-backend validation tests for model parity, sweeps, stimulation, delays, and feature pipelines.

The highest-priority simulator engineering rule is backend parity: the same spec should either produce the same contract across backends or raise an explicit unsupported-feature error.

## New Inference API

The public inference entry point is:

```python
from vbi.inference import SNPE, BoxUniform
```

The API is intentionally close to `sbi`:

```python
import numpy as np

from vbi.inference import SNPE, BoxUniform

prior = BoxUniform(
    low=np.array([-6.0]),
    high=np.array([-3.0]),
)

inference = SNPE(
    prior=prior,
    density_estimator="maf",
    backend="auto",
)

inference = inference.append_simulations(theta, x)
estimator = inference.train(training_batch_size=256)
posterior = inference.build_posterior(estimator, sample_with="direct")

samples = posterior.sample((1000,), x=x_obs)
log_probs = posterior.log_prob(theta, x=x_obs)
```

## Inference Development Highlights

The `vbi.inference` stack currently provides:

- `SNPE` with an `sbi`-compatible training flow;
- prior objects such as `BoxUniform`, `Gaussian`, `MultivariateNormal`, `LogNormal`, `Gamma`, `Beta`, and `MultipleIndependent`;
- neural density estimators including MDN, MAF, and NSF;
- NumPy/autograd and JAX backend paths;
- embedding-network support;
- posterior sampling with direct, rejection, and MCMC modes;
- Metropolis-Hastings and fixed-trajectory HMC posterior refinement;
- diagnostics including SBC, TARP, C2ST, pair plots, conditional pair plots, and loss plotting;
- compatibility utilities for converting simulator outputs into inference-ready arrays.

The intended direction is that `vbi.simulator` generates `(theta, x)` training data and `vbi.inference` trains the posterior:

```python
theta = values[:, :n_parameters]
x = values[:, n_parameters:]

inference = SNPE(prior=prior, density_estimator="nsf", backend="auto")
inference = inference.append_simulations(theta, x)
estimator = inference.train()
posterior = inference.build_posterior(estimator)
```

## End-to-End Direction

The long-term target is a clean workflow:

```python
# 1. Define simulator spec
spec = SimulationSpec(...)

# 2. Define parameter sweep
sweep_spec = SweepSpec(...)

# 3. Generate training data
labels, values = Sweeper(spec, sweep_spec, backend="numba").run(duration=5000.0)

# 4. Train posterior
inference = SNPE(prior=prior, density_estimator="maf", backend="auto")
inference = inference.append_simulations(theta, features)
estimator = inference.train()

# 5. Sample posterior for an observed feature vector
posterior = inference.build_posterior(estimator, sample_with="direct")
samples = posterior.sample((5000,), x=x_obs)
```

The simulator and inference modules should remain separate but composable:

- `vbi.simulator` owns model execution and feature generation.
- `vbi.inference` owns density estimation, posterior construction, posterior sampling, and diagnostics.

## Compatibility Commitments During Transition

During the transition, VBI should preserve these guarantees:

- Existing public examples should keep working until replacement examples are available.
- New simulator workflows should be validated against previous implementations before older paths are deprecated.
- Optional backends should remain optional; missing JAX, CUDA, C++, or GPU dependencies should not break basic CPU use.
- Backend-specific limitations should be documented or raised clearly.
- Root documentation should identify the recommended API without hiding compatibility paths.

## What Should Go Into The Root README Later

The final root `README.md` should be shorter than this draft. It should include:

- a short description of VBI;
- installation commands;
- a “Recommended API” section introducing `vbi.simulator` and `vbi.inference`;
- one minimal simulator example;
- one minimal inference example;
- a backend table;
- links to migration docs and examples;
- citation and contribution information.

Detailed backend reviews, milestone checklists, and implementation caveats should stay outside the root README.

## Suggested Release Note

This transition introduces a new unified simulator and inference stack. New projects should prefer `vbi.simulator` for model execution and `vbi.inference` for simulation-based inference. Existing model-specific APIs remain available during the migration period and will be replaced gradually after parity tests, examples, and migration guides are in place.
