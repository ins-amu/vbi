
# Virtual Brain Inference (VBI)
![logo](./vbi_logo.png "Title")
## installation

### Requirements

- Python3
- Matplotlib
- Scipy
- numpy
- C++ >= 11
- jidt [optional]  

```sh
conda env create --file environment.yml --name vbi
conda activate vbi

# gpu support
# conda install -c conda-forge cupy cudatoolkit=11.3
# conda install -c conda-forge pytorch-gpu

# If you need to use models implemented in C++ :
cd vbi/CPPModels
make  
# you need to install swig if you get an error and probably write the version of 
# python you are using at makefile
PYTHON_VERSION = 3.8 # or whatever version you have

# optional 
conda install -c conda-forge jpype1
```

you need to have install swig on you machine to compile C++ codes.

```sh
sudo apt-get install clang
sudo apt-get install swig
sudo apt-get install python3-dev # or [python3.9-dev] depends the default version of python on your machine.
# unless you get an error which says: fatal error,  Python.h not found.
```

- documentation:

```sh
cd docs 
doxygen Doxyfile
make latexpdf      # for pdf file
```
