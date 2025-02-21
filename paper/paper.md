---
title: 'Virtual Brain Inference (VBI): A Flexible and Integrative Toolkit'
tags:
  - Python
  - Neuroscience
  - Virtual Brain
  - Simulation-Based Inference
  - Probabilistic Inference
  - fMRI
  - EEG
  - MEG
authors:
  - name: Abolfazl Ziaeemehr
    orcid: 0000-0002-4696-9947
    affiliation: 1
    corresponding: true
  - name: Marmaduke Woodman
    orcid: 0000-0002-8410-4581
    affiliation: 1
  - name: Lia Domide
    orcid: 0000-0002-4822-2046
    affiliation: 2
  - name: Spase Petkoski
    orcid: 0000-0003-4540-6293
    affiliation: 1
  - name: Viktor Jirsa
    affiliation: 1
  - name: Meysam Hashemi
    orcid: 0000-0001-5289-9837
    affiliation: 1
    corresponding: true
affiliations:
  - name: Aix Marseille Univ, INSERM, INS, Institut de Neurosciences des Systèmes, Marseille, France
    index: 1
  - name: Codemart, Cluj-Napoca, Romania
    index: 2
date: 22 February 2025
bibliography: paper.bib
---

# Summary

Understanding complex brain dynamics and their neurobiological basis is a core challenge in neuroscience. Virtual brain modeling integrates computational models of brain dynamics with personalized anatomical data to address this challenge. Virtual Brain Inference (VBI) is a flexible, integrative Python toolkit designed for efficient probabilistic inference on virtual brain models. It fills a critical gap in tools for model inversion, enabling estimation of control parameters across neuroimaging modalities such as fMRI, EEG, and MEG. VBI provides fast simulations, feature extraction, efficient data storage, and probabilistic machine learning algorithms, delivering biophysically interpretable inferences from diverse recordings.

# Statement of Need

VBI is a Python-based toolkit tailored for probabilistic inference at the whole-brain scale. Combining Python’s flexibility with optimized C++ code for performance, VBI offers a user-friendly API supporting:

- **Brain models**: Wilson-Cowan, Montbrió, Jansen-Rit, Stuart-Landau, Wong-Wang, and Epileptor.
- **Fast simulation**: Just-in-time compilation of models across Python/C++ and CPU/GPU devices.
- **Feature extraction**: Functional connectivity (FC), functional connectivity dynamics (FCD), and power spectral density (PSD).
- **Deep neural density estimators**: Masked autoregressive flows (MAFs) and neural spline flows (NSFs).

VBI integrates structural and functional neuroimaging data, supporting space-efficient storage and memory-efficient batch processing. Traditional methods like Markov Chain Monte Carlo (MCMC) and Approximate Bayesian Computation (ABC) face significant challenges in this context. MCMC struggles with convergence in high-dimensional spaces and complex geometries, often requiring extensive tuning and computational resources. ABC, while likelihood-free, relies on predefined thresholds for sample acceptance, leading to inefficiencies and potential biases when rejecting samples that fall outside narrow criteria. In contrast, VBI leverages Simulation-Based Inference (SBI), which sidesteps these issues by using forward simulations and deep neural density estimators to directly approximate posterior distributions. This approach enhances efficiency, scalability, and robustness, making VBI particularly suited for inverting complex virtual brain models.

Designed for researchers and clinical applications, VBI enables personalized simulations of normal and pathological brain activity, aiding in distinguishing healthy from diseased states and informing targeted interventions. By addressing the inverse problem—estimating control parameters that best explain observed data—VBI leverages high-performance computing for parallel processing of large-scale datasets.


# Methods

Brain network dynamics at the regional scale are modeled as:

$$
\dot{\mathbf{\psi}}_i(t) = \mathcal{N}(\mathbf{\psi}_i) + G \sum_{j=1}^{N} \mathrm{w}_{ij} \mathcal{H}(\psi_i, \psi_j(t - \tau_{ij})) + z(\mathbf{\psi}_i) \xi_i(t),
$$

where $i = 1, 2, \ldots, N$, and $N$ is the number of brain regions. Here, $\mathbf{\psi}_i(t)$ represents local neural activity in region $i$, governed by the nonlinear function $\mathcal{N}(\mathbf{\psi}_i)$ when uncoupled. Interactions between regions are mediated by the coupling function $\mathcal{H}(\psi_i, \psi_j(t - \tau_{ij}))$, weighted by the structural connectivity $\mathrm{w}_{ij}$ (derived from diffusion-weighted MRI tractography) and delayed by axonal transmission times $\tau_{ij}$. A noise term, $z(\mathbf{\psi}_i) \xi_i(t)$, incorporates Gaussian noise with:

$$
\langle \xi_i(t) \rangle = 0, \quad \langle \xi_i(t) \xi_j(t') \rangle = 2 D \delta(t - t') \delta_{i,j},
$$

where $D$ is the noise strength. The system’s operating regime emerges from the interplay of global coupling $G$, local bifurcation parameters, and noise, with connectivity structure shaping macroscopic brain activity through delays and weights.

Simulation-based inference (SBI) in VBI avoids convergence issues of gradient-based MCMC methods and outperforms approximate Bayesian computation (ABC) by using deep neural density estimators to approximate posterior distributions, $p(\vec{\theta} \mid \vec{x}_{obs})$. SBI requires three components:

1. A prior distribution, $p(\vec{\theta})$, for sampling parameters $\vec{\theta}$.
2. A simulator, $p(\vec{x} \mid \vec{\theta})$, generating data $\vec{x}$ from parameters $\vec{\theta}$.
3. Low-dimensional features (e.g., FC, FCD, PSD) informative of target parameters.

These yield a training dataset $\{(\vec{\theta}_i, \vec{x}_i)\}_{i=1}^{N_{sim}}$, where $N_{sim}$ is the simulation budget. VBI uses MAFs or NSFs to learn the posterior distribution from observed data.

The VBI workflow comprises:

1. **Connectome construction**: From T1-weighted and diffusion-weighted MRI.
2. **Neural mass models**: Representing average neural population activity.
3. **Simulation**: Producing time series aligned with neuroimaging data.
4. **Feature extraction**: Computing FC, FCD, and PSD.
5. **Inference**: Training MAFs or NSFs to estimate posterior distributions.

### Evaluation of Posterior Fit

Posterior reliability is assessed using synthetic data via posterior z-scores ($z$) and shrinkage ($s$):

$$
z = \left| \frac{\bar{\theta} - \theta^\ast}{\sigma_{post}} \right|, \quad s = 1 - \frac{\sigma^2_{post}}{\sigma^2_{prior}},
$$

where $\bar{\theta}$ is the posterior mean, $\theta^\ast$ is the true parameter, and $\sigma_{post}$ and $\sigma_{prior}$ are posterior and prior standard deviations, respectively. High shrinkage indicates well-identified posteriors, while low z-scores confirm accurate capture of true values.

# Technical Terms

- **Control parameters**: Bifurcation parameters in a generative model that govern data synthesis and may reflect causal relationships.
- **Generative model**: A model (statistical, machine learning, or mechanistic) that generates data mimicking the original distribution.
- **Simulation-based inference**: A likelihood-free approach using forward simulations for inference in complex systems.
- **Virtual brain models**: Computational models of regional brain dynamics linked by a personalized structural connectivity matrix.

![Figure 1: Overview of the VBI workflow, integrating connectome construction, simulation, and inference.](Fig1.png)

# Acknowledgements

This research has received funding from EU’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreements No. 101147319 (EBRAINS 2.0 Project), No. 101137289 (Virtual Brain Twin Project), No. 101057429 (project environMENTAL), and government grant managed by the Agence Nationale de la Recherche reference ANR-22-PESN-0012 (France 2030 program). We acknowledge the use of Fenix Infrastructure resources, which are partially funded from the European Union’s Horizon 2020 research and innovation programme through the ICEI project under the grant agreement No. 800858. The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.

# References

