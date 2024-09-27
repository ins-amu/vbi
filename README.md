# Virtual Brain Inference (VBI)
<p align="center">
<img src="https://github.com/Ziaeemehr/vbi_paper/blob/main/vbi_log.png"  width="250">
</p>

## installation

```bash
    conda env create --file environment.yml --name vbi python=3.11
    conda activate vbi
    git clone https://github.com/Ziaeemehr/vbi.git
    cd vbi
    pip install .
    # pip install -e .[dev,docs] # with all depencendies
```

### Optional dependencies

`swig` need to be installed for using models implemented in C++ .

```bash
    sudo apt-get install swig
    sudo apt-get install python3-dev 
```
