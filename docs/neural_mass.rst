
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

