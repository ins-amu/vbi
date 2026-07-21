# Milestone: TVB-Style Jansen-Rit Coupling

## Goal

Update the modern simulator Jansen-Rit implementation so long-range coupling follows the TVB/Jansen-Rit interpretation while preserving TVB parameter names and scales.

The desired network input is:

```text
network_i(t) = G * sum_j W_ij * S(y1_j(t - tau_ji) - y2_j(t - tau_ji))
```

where:

```text
S(v) = 2 * nu_max / (1 + exp(r * (v0 - v)))
```

This means the sigmoid converts each source node's local PSP difference into a firing-rate drive before structural weighting and summation.

## Current Issue

The current modern `ModelSpec` path uses `CouplingSpec("difference")` to compute:

```text
c_y1_i = sum_j W_ij * (y1_j - y2_j)
```

and then applies the JR sigmoid inside the model equation:

```text
G * S(c_y1_i)
```

That is:

```text
G * S(W @ (y1 - y2))
```

This is not equivalent to TVB/JR-style coupling because `S` is nonlinear.

The desired form is:

```text
G * W @ S(y1 - y2)
```

For delayed coupling:

```text
G * sum_j W_ij * S(y1_j(t - tau_ji) - y2_j(t - tau_ji))
```

## Parameter Convention

Keep the TVB-style JR parameter names and scales:

```text
A, B, a, b, v0, nu_max, r, J, a_1, a_2, a_3, a_4, mu, G
```

Keep local JR population terms unchanged:

```text
S(y1_i - y2_i)
S(a_1 * J * y0_i)
S(a_3 * J * y0_i)
```

Only change the long-range network coupling placement of `S`.

## Target Equation

The `y4` equation should become:

```text
dy4_i = A*a * (
    mu
    + a_2*J*S(a_1*J*y0_i)
    + G * c_y1_i
) - 2*a*y4_i - a^2*y1_i
```

where the coupling backend provides:

```text
c_y1_i = sum_j W_ij * S(y1_j - y2_j)
```

or, with delays:

```text
c_y1_i = sum_j W_ij * S(y1_j(t - tau_ji) - y2_j(t - tau_ji))
```

## Implementation Tasks

1. Add a JR-specific coupling mode.

   Candidate names:

   ```python
   CouplingSpec("jr")
   CouplingSpec("jr_sigmoidal")
   CouplingSpec("psp_sigmoid")
   ```

   The mode should require exactly two coupling variables, `y1` and `y2`, and compute `W @ S(y1 - y2)`.

2. Update numpy coupling.

   File:

   ```text
   vbi/simulator/backend/numpy_/coupling.py
   ```

   Instant path:

   ```text
   c[0, tgt] = sum_src W[tgt, src] * S(y1[src] - y2[src])
   c[1, tgt] = 0
   ```

   Delayed path:

   ```text
   c[0, tgt] = sum_src W[tgt, src] *
               S(y1[src, t - tau(src,tgt)] - y2[src, t - tau(src,tgt)])
   c[1, tgt] = 0
   ```

3. Update numba coupling.

   File:

   ```text
   vbi/simulator/backend/numba_/_nb_sim.py
   ```

   Add instant and delayed kernels equivalent to the numpy implementation. The kernels need access to JR sigmoid parameters:

   ```text
   nu_max, r, v0
   ```

   Prefer passing these from model parameters rather than hard-coding defaults.

4. Update coupling spec validation.

   File:

   ```text
   vbi/simulator/spec/coupling.py
   ```

   Include the new coupling kind in the allowed literals and make sure unsupported backends fail clearly.

5. Update Jansen-Rit `ModelSpec`.

   File:

   ```text
   vbi/simulator/models/jansen_rit.py
   ```

   Change the long-range term from:

   ```text
   G * S(c_y1)
   ```

   to:

   ```text
   G * c_y1
   ```

   Update comments, LaTeX, and notes to say that `c_y1` is already the weighted incoming firing-rate drive.

6. Update workflow.

   File:

   ```text
   docs/examples/workflows/jr_vbi.py
   ```

   Use the new coupling mode:

   ```python
   coupling = CouplingSpec("jr")
   ```

   Also clean current workflow drift:

   ```text
   G_TRUE comment says 1.5, code uses 0.5
   docstring says t_cut=500 ms, code uses 0.0 ms
   docstring says duration=2501 ms, code uses 2500 ms
   exit(0) stops the SBI workflow before inference
   ```

## Delay Orientation

The simulator convention is:

```text
weights[tgt, src]
delay_steps[src, tgt]
```

The delayed JR coupling should read the source state with the delay from `src` to `tgt`, then accumulate into target:

```text
sum_src weights[tgt, src] * S(source_state[src, t - delay_steps[src, tgt]])
```

If tract-length matrices are symmetric, this is practically insensitive to orientation. If they are directed/asymmetric, the workflow and docs should make this convention explicit.

## Validation Plan

Add focused tests for:

1. Instant hand-computed coupling.

   For a small two- or three-node system, assert:

   ```text
   c_y1 == W @ S(y1 - y2)
   ```

2. Delayed hand-computed coupling.

   Build a small history buffer with known values and assert:

   ```text
   c_y1_i == sum_j W_ij * S(y1_j(t - tau_ji) - y2_j(t - tau_ji))
   ```

3. Numpy vs numba parity.

   Run JR with the new coupling mode for:

   ```text
   no delays
   nonzero delays
   euler
   heun
   ```

   Assert trajectories match within existing backend tolerances.

4. Zero-delay equivalence.

   With all tract lengths zero, the delayed and instant paths should produce the same coupling values.

5. Regression against older C++ JR convention.

   The older C++ JR code computes:

   ```text
   sum_j W_ij * S(y1_j - y2_j)
   ```

   Use it as a semantic reference for the network coupling term, while preserving the modern simulator backend APIs.

## Acceptance Criteria

- JR long-range coupling is `G * W @ S(y1 - y2)`.
- Delayed JR coupling is `G * W @ S(y1(t - tau) - y2(t - tau))`.
- `G` stays outside the sigmoid.
- TVB-style JR parameters are preserved.
- Numpy and numba backends agree for instant and delayed JR runs.
- Workflow uses the new coupling mode and no longer exits before inference unless explicitly intended.
- Run after edits:

```bash
graphify update .
```
