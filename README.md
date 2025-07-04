[![Test](https://github.com/ins-amu/vbi/actions/workflows/tests.yml/badge.svg)](https://github.com/ins-amu/vbi/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/vbi/badge/?version=latest)](https://vbi.readthedocs.io/latest/)
[![DOI](https://zenodo.org/badge/681090816.svg)](https://doi.org/10.5281/zenodo.14795543)
[![Docker Build](https://github.com/ins-amu/vbi/actions/workflows/docker-image.yml/badge.svg)](https://github.com/ins-amu/vbi/actions/workflows/docker-image.yml)
<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ins-amu/vbi/main?labpath=docs/examples/intro.ipynb) -->


# VBI: Virtual Brain Inference
---
[Getting Started](https://github.com/ins-amu/vbi/tree/main/docs/examples) |
[Documentation](https://vbi.readthedocs.io/latest/) | 

<p align="center">
<img src="https://github.com/Ziaeemehr/vbi_paper/blob/main/vbi_log.png"  width="250">
</p>


## installation

```bash
    conda env create --name vbi python=3.10
    conda activate vbi
    # from pip: Recommended
    pip install vbi
    # from source: More recent update
    git clone https://github.com/ins-amu/vbi.git
    cd vbi
    pip install .

    # pip install -e .[all,dev,docs]
    
    # To skip C++ compilation, use the following environment variable and install from source:
    SKIP_CPP=1 pip install -e . 
```

## Using Docker

To use the Docker image, you can pull it from the GitHub Container Registry and run it as follows:

```bash
    # Get it without building anything locally
    # without GPU
    docker run --rm -it -p 8888:8888 ghcr.io/ins-amu/vbi:main

    # with GPU
    docker run --gpus all --rm -it -p 8888:8888 ghcr.io/ins-amu/vbi:main


    # or build it locally:
    docker build -t vbi-project .                      # build
    docker run --gpus all -it -p 8888:8888 vbi-project # use with gpu

```    

- Quick check :

```python

    import vbi
    vbi.tests()
    vbi.test_imports()

    
  #              Dependency Check              
  #                                         
  #  Package      Version       Status        
  #━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
  #  vbi          v0.1.3        ✅ Available  
  #  numpy        1.24.4        ✅ Available  
  #  scipy        1.10.1        ✅ Available  
  #  matplotlib   3.7.5         ✅ Available  
  #  sbi          0.22.0        ✅ Available  
  #  torch        2.4.1+cu121   ✅ Available  
  #  cupy         12.3.0        ✅ Available  
  #                                          
  #  Torch GPU available: True
  #  Torch device count: 1
  #  Torch CUDA version: 12.1
  #  CuPy GPU available: True
  #  CuPy device count: 1
  #  CUDA Version: 11.8
  #  Device Name: NVIDIA RTX A5000
  #  Total Memory: 23.68 GB
  #  Compute Capability: 8.6

```


## Feedback and Contributions

We welcome contributions to the VBI project! If you have suggestions, bug reports, or feature requests, please open an issue on our [GitHub repository](https://github.com/ins-amu/vbi/issues). To contribute code, fork the repository, create a new branch for your feature or bugfix, and submit a pull request. Make sure to follow our coding standards and include tests for your changes. For detailed guidelines, please refer to our [CONTRIBUTING.md](https://github.com/ins-amu/vbi/blob/main/CONTRIBUTING.md) file. Thank you for helping us improve VBI!


## Citation

```bibtex
@article{VBI, 
title={Virtual Brain Inference (VBI): A flexible and integrative toolkit for efficient probabilistic inference on virtual brain models},
author={Ziaeemehr, Abolfazl and Woodman, Marmaduke and Domide, Lia and Petkoski, Spase and Jirsa, Viktor and Hashemi, Meysam},
DOI={10.7554/elife.106194.1}, 
url={http://dx.doi.org/10.7554/eLife.106194.1}, 
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

This research has received funding from:

- EU's Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreements:
  - No. 101147319 (EBRAINS 2.0 Project)
  - No. 101137289 (Virtual Brain Twin Project)
  - No. 101057429 (project environMENTAL)
- Government grant managed by the Agence Nationale de la Recherche:
  - Reference ANR-22-PESN-0012 (France 2030 program)

The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript.
