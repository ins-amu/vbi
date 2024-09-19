Virtual Brain Inference (VBI)
##############################


Installation
============

.. code-block:: bash


    conda env create --file environment.yml --name vbi python=3.11
    conda activate vbi
    git clone https://github.com/Ziaeemehr/vbi.git
    cd vbi
    pip install .



`swig` the models implemented in C++ (optional).

.. code-block:: bash

    sudo apt-get install swig
    sudo apt-get install python3-dev 


Examples
=========

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   examples/intro
   examples/intro_feature
   examples/do_cpp
   examples/do_nb
   examples/vep_sde


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`






