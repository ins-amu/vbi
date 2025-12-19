.. raw:: html

   <link rel="stylesheet" type="text/css" href="_static/custom.css">


Virtual Brain Inference (VBI)
##############################


.. image:: _static/vbi_log.png
   :alt: VBI Logo
   :width: 200px
   :align: center


The **Virtual Brain Inference (VBI)** toolkit is an open-source, flexible solution tailored for probabilistic inference on virtual brain models. It integrates computational models with personalized anatomical data to deepen the understanding of brain dynamics and neurological processes. VBI supports **fast simulations**, comprehensive **feature extraction**, and employs **deep neural density estimators** to handle various neuroimaging data types. Its goal is to bridge the gap in solving the inverse problem of identifying control parameters that best explain observed data, thereby making these models applicable for clinical settings. VBI leverages high-performance computing through GPU acceleration and C++ code to ensure efficiency in processing.


Workflow
========

.. image:: _static/Fig1.png
   :alt: VBI Logo
   :width: 800px

Installation
============

**Quick Start:**

First, create a conda environment:

.. code-block:: bash

    conda create --name vbi python=3.10
    conda activate vbi

Then install VBI:

.. code-block:: bash

    pip install vbi                    # Light version (CPU only)
    pip install vbi[inference]         # With parameter inference  
    pip install vbi[inference-gpu]     # Full functionality with GPU

**Verify Installation:**

.. code-block:: python 

   import vbi
   vbi.tests()
   vbi.test_imports()

**Complete Installation Guide:**

For detailed instructions including Docker, platform-specific guides (Windows, Google Colab, EBRAINS), troubleshooting, and installation from source, see our comprehensive :doc:`installation` guide.

   import vbi 
   vbi.tests()
   vbi.test_imports()  

**Example output for full installation:**

.. code-block:: text

   Dependency Check              
                                           
   Package      Version       Status        
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
   vbi          v0.2.1        ✅ Available  
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

**Example output for light version:**

.. code-block:: text

   Dependency Check              
                                           
   Package      Version       Status        
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
   vbi          v0.2.1        ✅ Available  
   numpy        1.24.4        ✅ Available  
   scipy        1.10.1        ✅ Available  
   matplotlib   3.7.5         ✅ Available  
   sbi          -             ❌ Not Found  
   torch        -             ❌ Not Found  
   cupy         -             ❌ Not Found  

   Note: Missing packages are expected for light installation.
   Install vbi[inference] or vbi[inference-gpu] for additional functionality.

Interactive Examples
====================

Try these examples directly in your browser using Google Colab:

.. list-table:: Colab Examples
   :header-rows: 1
   :class: color-caption

   * - **Model**
     - **Colab Link**
   * - Jansen-Rit SDE
     - |colab1|
   * - Wilson-Cowan SDE
     - |colab2|
   * - Wong-Wang SDE
     - |colab3|
   * - VEP SDE (84 Regions)
     - |colab4|
   * - VEP SDE (few parameters)
     - |colab5|
   * - MPR SDE
     - |colab6|
   * - Damped Oscillator
     - |colab7|

.. |colab1| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/jansen_rit_sde_numba_cde.ipynb
   :alt: Open In Colab

.. |colab2| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/wilson_cowan_sde_numba_cde.ipynb
   :alt: Open In Colab

.. |colab3| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/ww_full_sde_numba_cde.ipynb
   :alt: Open In Colab

.. |colab4| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/vep_sde_numba_cde_84.ipynb
   :alt: Open In Colab

.. |colab5| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/vep_sde_numba_cde.ipynb
   :alt: Open In Colab

.. |colab6| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/mpr_sde_numba_cde.ipynb
   :alt: Open In Colab

.. |colab7| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/Ziaeemehr/vbi_paper/blob/main/docs/examples/damp_oscillator_cde.ipynb
   :alt: Open In Colab

Documentation
=============

Getting Started
---------------

.. toctree::
   :maxdepth: 1

   installation
   docker_build
   docker_quickstart

Publications & Citations
------------------------

.. toctree::
   :maxdepth: 1

   publication

Brain Models
------------

.. toctree::
   :maxdepth: 1

   models

Inference Methods
-----------------

.. toctree::
   :maxdepth: 1

   inference_examples
   inference_cde_guide
   inference_cde_mdn_basic

Examples & Tutorials
--------------------

.. toctree::
   :maxdepth: 1

   examples_overview
   Examples

API Reference
-------------

.. toctree::
    :maxdepth: 1

    API


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



