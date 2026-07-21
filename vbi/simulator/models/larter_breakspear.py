from vbi.simulator.spec.model import ModelSpec, StateVar, Parameter

# All five intermediates (m_Ca, m_Na, m_K, QV, QZ) are inlined so
# the dfun_str expressions contain only state-vars, parameters and coupling.
# TVB parameter 'b' is retained; it does not conflict with the coupling
# alias 'c' (parameter 'C' is capital, also no conflict).

larter_breakspear = ModelSpec(
    name="LarterBreakspear",
    state_variables=(
        StateVar("V", default_init=0.0,  noise=True, lower_bound=-1.5, upper_bound=1.5),
        StateVar("W", default_init=0.0,  noise=False, lower_bound=-1.5, upper_bound=1.5),
        StateVar("Z", default_init=0.0,  noise=False, lower_bound=-1.5, upper_bound=1.5),
    ),
    parameters=(
        Parameter("gCa",   1.1,    "Ca channel conductance"),
        Parameter("gK",    2.0,    "K channel conductance"),
        Parameter("gL",    0.5,    "Leak channel conductance"),
        Parameter("phi",   0.7,    "Temperature scaling factor for K gate"),
        Parameter("gNa",   6.7,    "Na channel conductance"),
        Parameter("TK",    0.0,    "K channel half-activation voltage"),
        Parameter("TCa",  -0.01,   "Ca channel half-activation voltage"),
        Parameter("TNa",   0.3,    "Na channel half-activation voltage"),
        Parameter("VCa",   1.0,    "Ca reversal (Nernst) potential"),
        Parameter("VK",   -0.7,    "K reversal potential"),
        Parameter("VL",   -0.5,    "Leak reversal potential"),
        Parameter("VNa",   0.53,   "Na reversal potential"),
        Parameter("d_K",   0.3,    "K half-activation width"),
        Parameter("tau_K", 1.0,    "K relaxation time constant (ms)"),
        Parameter("d_Na",  0.15,   "Na half-activation width"),
        Parameter("d_Ca",  0.15,   "Ca half-activation width"),
        Parameter("aei",   2.0,    "Excitatory→inhibitory coupling strength"),
        Parameter("aie",   2.0,    "Inhibitory→excitatory coupling strength"),
        Parameter("b",     0.1,    "Time constant scaling for Z equation"),
        Parameter("C",     0.1,    "Long-range excitatory coupling weight (0 < C < a_ee)"),
        Parameter("ane",   1.0,    "Non-specific→excitatory input strength"),
        Parameter("ani",   0.4,    "Non-specific→inhibitory input strength"),
        Parameter("aee",   0.4,    "Excitatory→excitatory coupling strength"),
        Parameter("Iext",  0.3,    "Subcortical (thalamic) input current"),
        Parameter("rNMDA", 0.25,   "NMDA/AMPA receptor ratio"),
        Parameter("VT",    0.0,    "Excitatory firing threshold"),
        Parameter("d_V",   0.65,   "Excitatory threshold variance"),
        Parameter("ZT",    0.0,    "Inhibitory firing threshold"),
        Parameter("d_Z",   0.7,    "Inhibitory threshold variance"),
        Parameter("QV_max", 1.0,   "Maximum excitatory firing rate"),
        Parameter("QZ_max", 1.0,   "Maximum inhibitory firing rate"),
        Parameter("t_scale", 1.0,  "Time scale factor"),
    ),
    cvar=("V",),
    dfun_str={
        # Inlined intermediates:
        #   m_Ca = 0.5*(1+tanh((V-TCa)/d_Ca))
        #   m_Na = 0.5*(1+tanh((V-TNa)/d_Na))
        #   m_K  = 0.5*(1+tanh((V-TK)/d_K))
        #   QV   = 0.5*QV_max*(1+tanh((V-VT)/d_V))
        #   QZ   = 0.5*QZ_max*(1+tanh((Z-ZT)/d_Z))
        "V": (
            "t_scale * ("
            "-(gCa + (1.0-C)*rNMDA*aee*(0.5*QV_max*(1+tanh((V-VT)/d_V))) + C*rNMDA*aee*c)"
            "*(0.5*(1+tanh((V-TCa)/d_Ca)))*(V-VCa)"
            " - gK*W*(V-VK)"
            " - gL*(V-VL)"
            " - (gNa*(0.5*(1+tanh((V-TNa)/d_Na)))"
            "   + (1.0-C)*aee*(0.5*QV_max*(1+tanh((V-VT)/d_V)))"
            "   + C*aee*c"
            "  )*(V-VNa)"
            " - aie*Z*(0.5*QZ_max*(1+tanh((Z-ZT)/d_Z)))"
            " + ane*Iext"
            ")"
        ),
        "W": "t_scale * phi * (0.5*(1+tanh((V-TK)/d_K)) - W) / tau_K",
        "Z": "t_scale * b * (ani*Iext + aei*V*(0.5*QV_max*(1+tanh((V-VT)/d_V))))",
    },
    noise_variables=("V",),
    reference=(
        "Larter R et al. A coupled ODE lattice model for the simulation of "
        "epileptic seizures. Chaos 9(3):795, 1999. "
        "Breakspear M et al. Network: Computation in Neural Systems 14:703-732, 2003."
    ),
    dfun_latex={
        "V": (
            r"t_s\!\left["
            r"-(g_{\rm Ca} + (1{-}C)\,r_{\rm NMDA}\,a_{ee}\,Q_V + C\,r_{\rm NMDA}\,a_{ee}\,c)\,m_{\rm Ca}(V-V_{\rm Ca})"
            r" - g_K W(V-V_K) - g_L(V-V_L)"
            r" - (g_{\rm Na}\,m_{\rm Na} + (1{-}C)\,a_{ee}\,Q_V + C\,a_{ee}\,c)(V-V_{\rm Na})"
            r" - a_{ie}\,Z\,Q_Z + a_{ne}\,I_{\rm ext}"
            r"\right]"
        ),
        "W": r"t_s\,\phi\,\frac{m_K - W}{\tau_K}",
        "Z": r"t_s\,b\!\left(a_{ni}\,I_{\rm ext} + a_{ei}\,V\,Q_V\right)",
    },
    latex_notes=(
        r"Gate variables: $m_X = \frac{1}{2}\!\left(1+\tanh\frac{V-T_X}{\delta_X}\right)$ for $X\in\{{\rm Ca,Na,K}\}$. "
        r"Firing rates: $Q_V = \frac{1}{2}Q_V^{\max}\!\left(1+\tanh\frac{V-V_T}{\delta_V}\right)$, "
        r"$Q_Z = \frac{1}{2}Q_Z^{\max}\!\left(1+\tanh\frac{Z-Z_T}{\delta_Z}\right)$. "
        r"$C$ scales the long-range coupling $c^{\rm net}$; $C=0$ → no LRC."
    ),
)
