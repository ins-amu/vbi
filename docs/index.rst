Virtual Brain Inference (VBI)
##############################


Installation
============

Requirements
============

- Python3
- Matplotlib
- Scipy
- numpy
- C++ >= 11

.. code-block:: bash


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


`swig` need to be installed for using models implemented in C++ .

.. code-block:: bash

    sudo apt-get install swig
    sudo apt-get install python3-dev # or [python3.9-dev] depends the default version of python on your machine.
    # unless you get an error which says: fatal error,  Python.h not found.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Modules
########

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   neural_mass

