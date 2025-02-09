[![Test](https://github.com/ins-amu/vbi/actions/workflows/tests.yml/badge.svg)](https://github.com/ins-amu/vbi/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/vbi/badge/?version=latest)](https://vbi.readthedocs.io/latest/)
[![DOI](https://zenodo.org/badge/681090816.svg)](https://doi.org/10.5281/zenodo.14795543)



# Virtual Brain Inference (VBI)
<p align="center">
<img src="https://github.com/Ziaeemehr/vbi_paper/blob/main/vbi_log.png"  width="250">
</p>


## installation

```bash
    conda env create --name vbi python=3.10
    conda activate vbi
    git clone https://github.com/ins-amu/vbi.git
    cd vbi
    pip install .

    # pip install -e .[all,dev,docs]
```

## Using Docker

To use the Docker image, you can pull it from the GitHub Container Registry and run it as follows:

```bash
    docker run --rm -it ghcr.io/ins-amu/vbi:main python3 -c 'from vbi.utils import test_imports; test_imports()'

    # if gpu available:
    docker run --rm -it --gpus all -e GPU_SUPPORT=true ghcr.io/ins-amu/vbi:main python3 -c 'from vbi.utils import test_imports; test_imports()'
```


## Citation

```bibtex
@article{VBI,
  author = {Ziaeemehr, Abolfazl and Woodman, Marmaduke and Domide, Lia and Petkoski, Spase and Jirsa, Viktor and Hashemi, Meysam},
  title = {Virtual Brain Inference (VBI): A flexible and integrative toolkit for efficient probabilistic inference on virtual brain models},
  journal = {bioRxiv},
  year = {2025},
  doi = {10.1101/2025.01.21.633922},
  url = {https://doi.org/10.1101/2025.01.21.633922},
  abstract = {Network neuroscience has proven essential for understanding the principles and mechanisms underlying complex brain (dys)function and cognition. In this context, whole-brain network modeling--also known as virtual brain modeling--combines computational models of brain dynamics (placed at each network node) with individual brain imaging data (to coordinate and connect the nodes), advancing our understanding of the complex dynamics of the brain and its neurobiological underpinnings. However, there remains a critical need for automated model inversion tools to estimate control (bifurcation) parameters at large scales and across neuroimaging modalities, given their varying spatio-temporal resolutions. This study aims to address this gap by introducing a flexible and integrative toolkit for efficient Bayesian inference on virtual brain models, called Virtual Brain Inference (VBI). This open-source toolkit provides fast simulations, taxonomy of feature extraction, efficient data storage and loading, and probabilistic machine learning algorithms, enabling biophysically interpretable inference from non-invasive and invasive recordings. Through in-silico testing, we demonstrate the accuracy and reliability of inference for commonly used whole-brain network models and their associated neuroimaging data. VBI shows potential to improve hypothesis evaluation in network neuroscience through uncertainty quantification, and contribute to advances in precision medicine by enhancing the predictive power of virtual brain models.}
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
