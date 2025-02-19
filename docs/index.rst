.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/custom.css">


Virtual Brain Inference (VBI)
##############################


.. image:: _static/vbi_log.png
   :alt: VBI Logo
   :width: 300px

Installation
============


.. code-block:: bash


    conda env create --name vbi python=3.10
    conda activate vbi
    git clone https://github.com/ins-amu/vbi.git
    cd vbi
    pip install .
    # pip install -e .[all,dev,docs]


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   models


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
   examples/mpr_sde_cupy
   examples/mpr_sde_numba
   examples/jansen_rit_sde_cpp
   examples/ww_sde_torch_kong
   examples/ghb_sde_cupy



.. toctree::
    :maxdepth: 2
    :caption: API Reference

    API


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



