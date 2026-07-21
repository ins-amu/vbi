[![Test](https://github.com/ins-amu/vbi/actions/workflows/tests.yml/badge.svg)](https://github.com/ins-amu/vbi/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/vbi/badge/?version=latest)](https://vbi.readthedocs.io/latest/)
[![DOI](https://zenodo.org/badge/681090816.svg)](https://doi.org/10.5281/zenodo.14795543)
[![Docker Build](https://github.com/ins-amu/vbi/actions/workflows/docker-image.yml/badge.svg)](https://github.com/ins-amu/vbi/actions/workflows/docker-image.yml)
<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ins-amu/vbi/main?labpath=examples/intro.ipynb) -->


# VBI: Virtual Brain Inference
---
[Getting Started](https://github.com/ins-amu/vbi/tree/main/examples) |
[Documentation](https://vbi.readthedocs.io/latest/) | 

<p align="center">
  <img src="https://raw.githubusercontent.com/ins-amu/vbi/main/vbi_log.png" width="400">
</p>


## Statement of Need

Whole-brain network modeling - also known as *virtual brain modeling* - combines
computational models of neural dynamics with individual structural connectivity derived
from neuroimaging data. A central challenge in this field is *model inversion*: given
observed recordings (EEG, MEG, fMRI, SEEG), how do we identify the set of control
parameters that best reproduces the measured brain dynamics? This inverse problem is
high-dimensional, computationally expensive, and subject to significant uncertainty,
making it inaccessible with traditional optimization or MCMC approaches at the scale
required for clinical translation.

**VBI addresses this gap** by providing a unified, open-source toolkit that integrates:

- **Fast simulations** via GPU-accelerated (CuPy), JIT-compiled (Numba/JAX), and C++ backends for a library of established whole-brain models (Jansen-Rit, Wilson-Cowan, Wong-Wang, MPR, VEP, and others);
- **Feature extraction** with a taxonomy of summary statistics linking simulated and empirical neuroimaging data;
- **Simulation-based inference (SBI)** using deep neural density estimators (via the `sbi` package) to obtain full posterior distributions over model parameters.

The primary audience is computational neuroscientists and neuroimaging researchers who
work with personalized brain models, as well as clinical researchers exploring virtual
brain twins for precision medicine applications (e.g., epilepsy, multiple sclerosis,
Parkinson's disease). While general-purpose SBI libraries such as
[sbi](https://sbi-dev.github.io/sbi/) and simulation frameworks such as
[The Virtual Brain (TVB)](https://www.thevirtualbrain.org) exist independently, VBI
provides the end-to-end pipeline - from biophysically realistic simulation to
probabilistic parameter estimation - specifically designed and validated for whole-brain
network models, substantially lowering the barrier to applying these methods in
neuroscience research.

## Installation

### Quick Start

```bash
# Create conda environment (recommended)
conda create --name vbi python=3.10
conda activate vbi


# Install VBI
export SKIP_CPP=1                  # To skip C++ compilation
pip install vbi                    # Light version (CPU only)
pip install vbi[inference]         # With parameter inference
pip install vbi[inference-gpu]     # Full functionality with GPU support
```

👉 Ready to try it out? Start with a working example: [Introduction & Feature Extraction](https://vbi.readthedocs.io/latest/examples/intro_feature.html).

### Using Docker

```bash
# Quick start with pre-built image
docker run --rm -it -p 8888:8888 ghcr.io/ins-amu/vbi:main

# With GPU support
docker run --gpus all --rm -it -p 8888:8888 ghcr.io/ins-amu/vbi:main
```

### Other Installation Methods

For detailed installation instructions including:
- **Installing from source**
- **Windows-specific installation**  
- **Building Docker locally**
- **Platform-specific guides** (Google Colab, EBRAINS)
- **Troubleshooting**

See our comprehensive [Installation Guide](https://vbi.readthedocs.io/latest/installation.html) in the documentation.

### Quick Verification

```python
import vbi
vbi.tests()
vbi.test_imports()
```

**Example output:**
```
              Dependency Check              
                                         
  Package      Version       Status        
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  vbi          v0.4.3        ✅ Available  
  numpy        1.24.4        ✅ Available  
  scipy        1.10.1        ✅ Available  
  matplotlib   3.7.5         ✅ Available  
  sbi          0.22.0        ✅ Available  
  torch        2.4.1+cu121   ✅ Available  
  cupy         12.3.0        ✅ Available  
                                          
  Torch GPU available: True
  Torch device count: 1
  Torch CUDA version: 12.1
  CuPy GPU available: True
  CuPy device count: 1
```

## Getting Started

- **📚 [Documentation](https://vbi.readthedocs.io/latest/)** - Complete guides and API reference
- **🎯 [Examples](https://github.com/ins-amu/vbi/tree/main/examples)** - Jupyter notebooks with tutorials
- **🚀 [Quick Start](https://vbi.readthedocs.io/latest/examples_overview.html)** - Choose your computational backend

### Interactive Examples

Try these examples directly in your browser using Google Colab:

[Load simulated data](https://amubox.univ-amu.fr/s/oEdiaJSbjJcWfCP)

| Model | Colab Link |
|-------|------------|
| Jansen-Rit SDE | <a href="https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/jansen_rit_sde_numba_cde.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Wilson-Cowan SDE | <a href="https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/wilson_cowan_sde_numba_cde.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Wong-Wang SDE | <a href="https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/ww_full_sde_numba_cde.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| VEP SDE (84 Regions) | <a href="https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/vep_sde_numba_cde_84.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| VEP SDE (few parameters) | <a href="https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/vep_sde_numba_cde.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| MPR SDE | <a href="https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/mpr_sde_numba_cde.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Damped Oscillator | <a href="https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/damp_oscillator_cde.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## Upcoming: Unified API (v0.5)

VBI is transitioning to a unified `vbi.simulator` and `vbi.inference` API that works consistently across all computational backends (NumPy, Numba, CUDA, C++). The current model-specific paths remain fully supported during this transition.

> Documentation is in progress and will be published alongside the stable release. For the full transition plan, see [notes/README.md](notes/README.md).

## Feedback and Contributions

We welcome contributions to the VBI project! If you have suggestions, bug reports, or feature requests, please open an issue on our [GitHub repository](https://github.com/ins-amu/vbi/issues). To contribute code, fork the repository, create a new branch for your feature or bugfix, and submit a pull request. Make sure to follow our coding standards and include tests for your changes. For detailed guidelines, please refer to our [CONTRIBUTING.md](https://github.com/ins-amu/vbi/blob/main/CONTRIBUTING.md) file. Thank you for helping us improve VBI!


## GDPR Compliance

VBI itself does not collect, store, or transmit personal data.  
Users may provide their own datasets locally; in such cases, the responsibility for GDPR compliance lies with the user and their execution environment.  

For details, see [GDPR_COMPLIANCE.md](GDPR_COMPLIANCE.md).


## Citation

```bibtex
@article{VBI, 
title={Virtual Brain Inference (VBI): A flexible and integrative toolkit for efficient probabilistic inference on virtual brain models},
author={Ziaeemehr, Abolfazl and Woodman, Marmaduke and Domide, Lia and Petkoski, Spase and Jirsa, Viktor and Hashemi, Meysam},
DOI={10.7554/elife.106194.2}, 
url={http://dx.doi.org/10.7554/eLife.106194.2}, 
publisher={eLife Sciences Publications, Ltd}, 
year={2025}, 
abstract = {Network neuroscience has proven essential for understanding the principles and mechanisms
underlying complex brain (dys)function and cognition. In this context, whole-brain network modeling–
also known as virtual brain modeling–combines computational models of brain dynamics (placed at each network node)
with individual brain imaging data (to coordinate and connect the nodes), advancing our understanding of
the complex dynamics of the brain and its neurobiological underpinnings. However, there remains a critical
need for automated model inversion tools to estimate control (bifurcation) parameters at large scales
associated with neuroimaging modalities, given their varying spatio-temporal resolutions.
This study aims to address this gap by introducing a flexible and integrative toolkit for efficient Bayesian inference
on virtual brain models, called Virtual Brain Inference (VBI). This open-source toolkit provides fast simulations,
taxonomy of feature extraction, efficient data storage and loading, and probabilistic machine learning algorithms,
enabling biophysically interpretable inference from non-invasive and invasive recordings.
Through in-silico testing, we demonstrate the accuracy and reliability of inference for commonly used
whole-brain network models and their associated neuroimaging data. VBI shows potential to improve hypothesis
evaluation in network neuroscience through uncertainty quantification, and contribute to advances in precision
medicine by enhancing the predictive power of virtual brain models.}
}
```


See the VBI paper  [here](https://elifesciences.org/reviewed-preprints/106194#tab-content).


This research has received funding from:

- EU's Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreements:
  - No. 101147319 (EBRAINS 2.0 Project)
  - No. 101137289 (Virtual Brain Twin Project)
  - No. 101057429 (project environMENTAL)
- Government grant managed by the Agence Nationale de la Recherche:
  - Reference ANR-22-PESN-0012 (France 2030 program)

The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.
