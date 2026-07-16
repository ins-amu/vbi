Limitations and Scope
#####################

VBI is designed for amortized simulation-based inference (SBI) on whole-brain
network models. It is most useful when you need full posterior distributions
over a modest number of biophysically meaningful control parameters and can
afford an upfront simulation and training budget in exchange for fast, reusable
inference on new observations.

When to use VBI
===============

VBI is a good fit when:

- You work with whole-brain network models (e.g. Jansen-Rit, Wilson-Cowan,
  Wong-Wang, Montbrió/MPR, Stuart-Landau, Epileptor/VEP) and want to invert them
  against empirical neuroimaging data (fMRI, EEG, MEG, SEEG).
- You want a full posterior distribution over model parameters, including
  uncertainty, rather than a single point estimate.
- You plan to perform inference repeatedly on many subjects or observations,
  so the one-time cost of amortized training pays off.
- You can run a sufficiently large simulation budget on CPU or GPU.

Limitations
===========

Users should keep the following limitations in mind:

**Parameter identifiability.**
Some parameters are only jointly (not individually) identifiable. For example,
in the Wong-Wang model the global coupling ``G`` and synaptic coupling ``J`` are
structurally non-identifiable and yield a curved, degenerate posterior. This is
a property of the model and the data rather than of VBI itself, but it limits
how uniquely individual parameters can be recovered.

**Choice of data features.**
Inference quality depends strongly on selecting low-dimensional data features
that are informative about the target parameters. Functional connectivity alone
(FC/FCD) is often insufficient for estimating regional parameters;
spatio-temporal features are usually required.

**Noise sensitivity.**
High observational or dynamical noise can corrupt feature estimation and degrade
the resulting posterior.

**Simulation budget.**
There is no principled rule for the number of simulations required to train a
reliable estimator. Adequacy must be checked empirically, for example using
posterior z-scores and shrinkage or simulation-based calibration (SBC).

**Upfront computational cost.**
Single-round amortized training requires a substantial one-time compute
investment (roughly tens of minutes to several hours, depending on the model and
simulation budget) before inference becomes inexpensive.

**Preprocessing.**
Some statistical information (e.g. signal mean and variance) can be lost during
feature preprocessing, reducing the informativeness of the extracted features.

Further reading
===============

For a detailed validation study and an in-depth discussion of these points, see
`Ziaeemehr et al., eLife 2025 <https://elifesciences.org/articles/106194>`_.
