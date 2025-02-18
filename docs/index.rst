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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



API and documentation
=====================

C++ models
-------------------------

.. automodule:: vbi
    :members:
    :undoc-members:

vbi.models.cpp.jansen_rit
-------------------------
    .. automodule:: vbi.models.cpp.jansen_rit
        :members:
        :undoc-members:

vbi.models.cpp.km
-------------------------
    .. automodule:: vbi.models.cpp.km
        :members:
        :undoc-members:

vbi.models.cpp.mpr 
-------------------------
    .. automodule:: vbi.models.cpp.mpr
        :members:
        :undoc-members:

vbi.models.cpp.vep 
-------------------------
    .. automodule:: vbi.models.cpp.vep
        :members:
        :undoc-members:


vbi.models.cpp.wc 
-------------------------
    .. automodule:: vbi.models.cpp.wc
        :members:
        :undoc-members:


Cupy models 
-------------------------

vbi.models.cupy.mpr 
-------------------------
    .. automodule:: vbi.models.cupy.mpr
        :members:
        :undoc-members:

vbi.models.cupy.ghb
-------------------------
    .. automodule:: vbi.models.cupy.ghb
        :members:
        :undoc-members:

vbi.models.cupy.jansen_rit
----------------------------
    .. automodule:: vbi.models.cupy.jansen_rit
        :members:
        :undoc-members:

vbi.models.cupy.utils 
-------------------------
    .. automodule:: vbi.models.cupy.utils
        :members:
        :undoc-members:
