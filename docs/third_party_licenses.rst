Third-Party Libraries & Licenses
=================================

VBI builds on a number of open-source libraries. We gratefully acknowledge their
authors and list each library, its license type, and a link to the full license
text below.

Core Dependencies
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Library
     - License
     - License URL
   * - `NumPy <https://numpy.org>`_
     - BSD-3-Clause
     - https://github.com/numpy/numpy/blob/main/LICENSE.txt
   * - `SciPy <https://scipy.org>`_
     - BSD-3-Clause
     - https://github.com/scipy/scipy/blob/main/LICENSE.txt
   * - `Matplotlib <https://matplotlib.org>`_
     - PSF / BSD-compatible
     - https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE
   * - `Numba <https://numba.pydata.org>`_
     - BSD-2-Clause
     - https://github.com/numba/numba/blob/main/LICENSE
   * - `pandas <https://pandas.pydata.org>`_
     - BSD-3-Clause
     - https://github.com/pandas-dev/pandas/blob/main/LICENSE
   * - `h5py <https://www.h5py.org>`_
     - BSD-3-Clause
     - https://github.com/h5py/h5py/blob/master/LICENSE
   * - `NetworkX <https://networkx.org>`_
     - BSD-3-Clause
     - https://github.com/networkx/networkx/blob/main/LICENSE.txt
   * - `scikit-learn <https://scikit-learn.org>`_
     - BSD-3-Clause
     - https://github.com/scikit-learn/scikit-learn/blob/main/COPYING
   * - `autograd <https://github.com/HIPS/autograd>`_
     - MIT
     - https://github.com/HIPS/autograd/blob/master/license.txt
   * - `tqdm <https://tqdm.github.io>`_
     - MIT / MPL-2.0
     - https://github.com/tqdm/tqdm/blob/master/LICENCE
   * - `Rich <https://rich.readthedocs.io>`_
     - MIT
     - https://github.com/Textualize/rich/blob/master/LICENSE
   * - `corner <https://corner.readthedocs.io>`_
     - MIT
     - https://github.com/dfm/corner.py/blob/main/LICENSE
   * - `nbconvert <https://nbconvert.readthedocs.io>`_
     - BSD-3-Clause
     - https://github.com/jupyter/nbconvert/blob/main/LICENSE
   * - `parameterized <https://github.com/wolever/parameterized>`_
     - BSD-2-Clause
     - https://github.com/wolever/parameterized/blob/master/LICENSE.txt
   * - `pytest <https://pytest.org>`_
     - MIT
     - https://github.com/pytest-dev/pytest/blob/main/LICENSE

Bundled Binary Dependencies
---------------------------

The following library is **bundled** inside the VBI package as a pre-compiled
JAR file (``vbi/feature_extraction/infodynamics.jar``) and is not installed
via pip.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Library
     - License
     - License URL
   * - `JIDT — Java Information Dynamics Toolkit <https://github.com/jlizier/jidt>`_
     - GPL-3.0
     - https://github.com/jlizier/jidt/blob/master/LICENCE.txt

JIDT is used for information-theoretic feature extraction (transfer entropy,
mutual information). It requires a Java JDK (>= 8) to be installed on the
host system; see the :doc:`installation` guide for details.

**GPL-3.0 isolation.** Because JIDT is GPL-3.0-licensed while VBI is
Apache-2.0-licensed, VBI never embeds JIDT's JVM inside its own process.
Instead, JIDT (via JPype) is run in a separate, independent OS process
(``vbi/feature_extraction/jidt_worker.py``), launched on demand and
communicating with the main VBI process only through a line-delimited
JSON protocol over stdin/stdout. The two programs exchange plain numeric
data across this narrow, well-defined interface rather than sharing an
address space or linking against one another, so the GPL-3.0 copyleft
terms that apply to the JIDT worker process do not extend to VBI's own
Apache-2.0-licensed codebase.

Optional / Inference Dependencies
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Library
     - License
     - License URL
   * - `PyTorch <https://pytorch.org>`_
     - BSD-3-Clause
     - https://github.com/pytorch/pytorch/blob/main/LICENSE
   * - `sbi <https://sbi-dev.github.io/sbi>`_
     - Apache-2.0
     - https://github.com/sbi-dev/sbi/blob/main/LICENSE.txt
   * - `JAX <https://jax.readthedocs.io>`_
     - Apache-2.0
     - https://github.com/google/jax/blob/main/LICENSE
   * - `jaxlib <https://jax.readthedocs.io>`_
     - Apache-2.0
     - https://github.com/google/jax/blob/main/LICENSE
   * - `CuPy <https://cupy.dev>`_
     - MIT
     - https://github.com/cupy/cupy/blob/main/LICENSE
   * - `pycatch22 <https://github.com/DynamicsAndNeuralSystems/pycatch22>`_
     - GPL-3.0
     - https://github.com/DynamicsAndNeuralSystems/pycatch22/blob/main/LICENSE

Documentation Dependencies
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Library
     - License
     - License URL
   * - `Sphinx <https://www.sphinx-doc.org>`_
     - BSD-2-Clause
     - https://github.com/sphinx-doc/sphinx/blob/master/LICENSE.rst
   * - `numpydoc <https://numpydoc.readthedocs.io>`_
     - BSD-2-Clause
     - https://github.com/numpy/numpydoc/blob/main/LICENSE.txt
   * - `nbformat <https://nbformat.readthedocs.io>`_
     - BSD-3-Clause
     - https://github.com/jupyter/nbformat/blob/main/LICENSE
   * - `nbsphinx <https://nbsphinx.readthedocs.io>`_
     - MIT
     - https://github.com/spatialaudio/nbsphinx/blob/master/LICENSE
   * - `Furo <https://pradyunsg.me/furo>`_
     - MIT
     - https://github.com/pradyunsg/furo/blob/main/LICENSE
   * - `pypandoc <https://github.com/JessicaTegner/pypandoc>`_
     - MIT
     - https://github.com/JessicaTegner/pypandoc/blob/master/LICENSE

Bundled Datasets
----------------

VBI ships a small sample structural connectivity dataset
(``vbi/dataset/connectivity_84``) used by the tutorials and examples. See
``vbi/dataset/README.md`` for full provenance details.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Dataset
     - License
     - Source
   * - connectivity_84 (Hagmann et al. 2008)
     - CC BY 3.0
     - Redistributed from `The Virtual Brain <https://github.com/the-virtual-brain/tvb-data>`_

Build Dependencies
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Library
     - License
     - License URL
   * - `setuptools <https://setuptools.pypa.io>`_
     - MIT
     - https://github.com/pypa/setuptools/blob/main/LICENSE
   * - `SWIG <https://www.swig.org>`_
     - GPL-3.0 with runtime exception
     - https://github.com/swig/swig/blob/master/LICENSE
   * - `Hatchling <https://hatch.pypa.io>`_
     - MIT
     - https://github.com/pypa/hatch/blob/master/LICENSE.txt
