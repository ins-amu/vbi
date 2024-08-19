
Mass Models
############


C++ implementation [CPU]
=========================

Damp Oscillator
---------------

.. automodule:: vbi.models.cpp.damp_oscillator
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Jansen-Rit Neural Mass Model
----------------------------

.. math::
    :nowrap:

    \begin{align*}
    \dot{y_0(t)} &= y_3(t) \\
    \dot{y_1(t)} &= y_4(t) \\
    \dot{y_2(t)} &= y_5(t) \\
    \dot{y_3(t)} &= Aa\, \sigma(y_1(t)-y_2(t)) - 2ay_3(t) - a^2 y_0(t)\\
    \dot{y_4(t)} &= Aa\Big[ P(t) +C_2 \, \sigma(C_1 y_0(t)) \Big] - 2a y_4(t) -a^2 y_1(t) \\
    \dot{y_5(t)} &= Bb \Big[ C_4\, \sigma(C_3 y_0(t)) \Big] -2by_5(t) -b^2 y_2(t) \\
    \sigma(v) &= \frac{v_{max}}{1+\exp(r(v_0-v))} \\
    \end{align*}


.. automodule:: vbi.models.cpp.jansen_rit
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:


MPR Model
---------------------

.. automodule:: vbi.models.cpp.mpr
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Kuramoto Model
---------------------

.. automodule:: vbi.models.cpp.km
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Stuart-Landau Model
---------------------

.. automodule:: vbi.models.cpp.sl
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

"""

Wong Wang model
---------------------

.. automodule:: vbi.models.cpp.ww
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:


Wilson-Cowan Model
---------------------

.. automodule:: vbi.models.cpp.wc
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Cupy implementation [CPU/GPU]
=============================

Generic-Hopf model
--------------------

.. automodule:: vbi.models.cupy.ghb
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Jansen-Rit Neural Mass Model
----------------------------

.. automodule:: vbi.models.cupy.jansen_rit
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Kuramoto Model
---------------------

.. automodule:: vbi.models.cupy.km
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

MPR Model
---------------------

.. automodule:: vbi.models.cupy.mpr
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Utils
--------

.. automodule:: vbi.models.cupy.utils
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

NUMBA implementation [CPU]
==========================

Damp Oscillator
---------------

.. automodule:: vbi.models.numba.damp_oscillator
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:



Generic-Hopf model
--------------------

.. automodule:: vbi.models.numba.ghb
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

MPR Model
---------------------

.. automodule:: vbi.models.numba.mpr
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:

Wang-Wong Model
---------------------

.. automodule:: vbi.models.numba.ww
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
