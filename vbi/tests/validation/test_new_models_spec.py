"""
Structural correctness tests for the new ModelSpec objects.

These tests do NOT run the simulator - they verify that the ModelSpec
fields are internally consistent: correct dimensions, valid indices,
matching keys, sensible bounds, etc.

Any model bug that manifests before a single integration step is caught here.
"""
from vbi.simulator.spec.connectivity import Connectivity
import pytest
import numpy as np

from vbi.simulator.models.generic_2d_oscillator import generic_2d_oscillator
from vbi.simulator.models.kuramoto import kuramoto
from vbi.simulator.models.sup_hopf import sup_hopf
from vbi.simulator.models.linear import linear
from vbi.simulator.models.larter_breakspear import larter_breakspear
from vbi.simulator.models.coombes_byrne_2d import coombes_byrne_2d
from vbi.simulator.models.gast_sd import gast_sd
from vbi.simulator.models.gast_sf import gast_sf

ALL_NEW_MODELS = [
    generic_2d_oscillator,
    kuramoto,
    sup_hopf,
    linear,
    larter_breakspear,
    coombes_byrne_2d,
    gast_sd,
    gast_sf,
]


# ---------------------------------------------------------------------------
# Expected metadata per model
# ---------------------------------------------------------------------------

MODEL_META = {
    "Generic2dOscillator": dict(
        n_sv=2, sv_names=("V", "W"), n_params=12,
        cvar=("V",), noise_vars=("V", "W"),
        dfun_keys={"V", "W"},
        bounded={"V": (-2.0, 4.0), "W": (-6.0, 6.0)},
    ),
    "Kuramoto": dict(
        n_sv=1, sv_names=("theta",), n_params=1,
        cvar=("theta",), noise_vars=("theta",),
        dfun_keys={"theta"},
        bounded={},
    ),
    "SupHopf": dict(
        n_sv=2, sv_names=("x", "y"), n_params=2,
        cvar=("x", "y"), noise_vars=("x", "y"),
        dfun_keys={"x", "y"},
        bounded={"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
    ),
    "Linear": dict(
        n_sv=1, sv_names=("x",), n_params=1,
        cvar=("x",), noise_vars=("x",),
        dfun_keys={"x"},
        bounded={"x": (-1.0, 1.0)},
    ),
    "LarterBreakspear": dict(
        n_sv=3, sv_names=("V", "W", "Z"), n_params=32,
        cvar=("V",), noise_vars=("V",),
        dfun_keys={"V", "W", "Z"},
        bounded={"V": (-1.5, 1.5), "W": (-1.5, 1.5), "Z": (-1.5, 1.5)},
    ),
    "CoombesByrne2D": dict(
        n_sv=2, sv_names=("r", "V"), n_params=4,
        cvar=("r", "V"), noise_vars=("r", "V"),
        dfun_keys={"r", "V"},
        bounded={"r": (0.0, None)},
    ),
    "GastSchmidtKnosche_SD": dict(
        n_sv=4, sv_names=("r", "V", "A", "B"), n_params=9,
        cvar=("r", "V"), noise_vars=("r", "V"),
        dfun_keys={"r", "V", "A", "B"},
        bounded={"r": (0.0, None)},
    ),
    "GastSchmidtKnosche_SF": dict(
        n_sv=4, sv_names=("r", "V", "A", "B"), n_params=9,
        cvar=("r", "V"), noise_vars=("r", "V"),
        dfun_keys={"r", "V", "A", "B"},
        bounded={"r": (0.0, None)},
    ),
}


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ALL_NEW_MODELS, ids=lambda m: m.name)
class TestModelSpecStructure:

    def test_name_nonempty(self, model):
        assert model.name and isinstance(model.name, str)

    def test_n_sv(self, model):
        meta = MODEL_META[model.name]
        assert model.n_sv == meta["n_sv"], \
            f"{model.name}: expected n_sv={meta['n_sv']}, got {model.n_sv}"

    def test_sv_names(self, model):
        meta = MODEL_META[model.name]
        assert model.sv_names == meta["sv_names"], \
            f"{model.name}: sv_names mismatch"

    def test_n_params(self, model):
        meta = MODEL_META[model.name]
        assert len(model.parameters) == meta["n_params"], \
            f"{model.name}: expected {meta['n_params']} params, got {len(model.parameters)}"

    def test_cvar_names(self, model):
        meta = MODEL_META[model.name]
        assert model.cvar == meta["cvar"], \
            f"{model.name}: cvar mismatch"

    def test_cvar_indices_in_range(self, model):
        for idx in model.cvar_indices:
            assert 0 <= idx < model.n_sv, \
                f"{model.name}: cvar index {idx} out of range [0, {model.n_sv})"

    def test_dfun_keys_match_sv_names(self, model):
        meta = MODEL_META[model.name]
        assert set(model.dfun_str.keys()) == meta["dfun_keys"], \
            f"{model.name}: dfun_str keys mismatch"

    def test_noise_variables_are_svs(self, model):
        for nv in model.noise_variables:
            assert nv in model.sv_names, \
                f"{model.name}: noise variable '{nv}' not in sv_names"

    def test_noise_indices_in_range(self, model):
        for idx in model.noise_indices:
            assert 0 <= idx < model.n_sv, \
                f"{model.name}: noise index {idx} out of range"

    def test_default_params_keys(self, model):
        defaults = model.default_params
        for p in model.parameters:
            assert p.name in defaults, \
                f"{model.name}: default missing for parameter '{p.name}'"

    def test_param_names_unique(self, model):
        names = model.param_names
        assert len(names) == len(set(names)), \
            f"{model.name}: duplicate parameter names"

    def test_sv_names_unique(self, model):
        assert len(model.sv_names) == len(set(model.sv_names)), \
            f"{model.name}: duplicate state variable names"

    def test_bounds_logical(self, model):
        for sv in model.state_variables:
            lo = sv.lower_bound
            hi = sv.upper_bound
            if lo is not None and hi is not None:
                assert lo < hi, \
                    f"{model.name}.{sv.name}: lower_bound ({lo}) >= upper_bound ({hi})"

    def test_bounds_match_expected(self, model):
        meta = MODEL_META[model.name]
        for sv in model.state_variables:
            if sv.name in meta["bounded"]:
                lo_exp, hi_exp = meta["bounded"][sv.name]
                if lo_exp is not None:
                    assert sv.lower_bound == lo_exp, \
                        f"{model.name}.{sv.name}: lower_bound should be {lo_exp}"
                if hi_exp is not None:
                    assert sv.upper_bound == hi_exp, \
                        f"{model.name}.{sv.name}: upper_bound should be {hi_exp}"

    def test_dfun_strings_nonempty(self, model):
        for sv_name, expr in model.dfun_str.items():
            assert expr.strip(), \
                f"{model.name}: dfun_str['{sv_name}'] is empty"

    def test_reference_nonempty(self, model):
        assert model.reference and isinstance(model.reference, str), \
            f"{model.name}: reference string missing"


# ---------------------------------------------------------------------------
# dfun expression safety: no reserved names used as parameters
# ---------------------------------------------------------------------------

RESERVED = {"c", "pi", "exp", "log", "sin", "cos", "tanh", "sqrt", "abs"}

@pytest.mark.parametrize("model", ALL_NEW_MODELS, ids=lambda m: m.name)
def test_no_reserved_param_names(model):
    """Parameter names must not collide with coupling aliases or math builtins."""
    for p in model.parameters:
        assert p.name not in RESERVED, \
            f"{model.name}: parameter '{p.name}' collides with reserved coupling/math name"
    # Also check c_<cvar> aliases
    for cname in model.cvar:
        alias = f"c_{cname}"
        for p in model.parameters:
            assert p.name != alias, \
                f"{model.name}: parameter '{p.name}' collides with coupling alias '{alias}'"


# ---------------------------------------------------------------------------
# dfun evaluation: dfun must produce finite output for default initial state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ALL_NEW_MODELS, ids=lambda m: m.name)
def test_dfun_evaluates_finite_at_defaults(model):
    """dfun evaluated at default_init with zero coupling must return finite values."""
    from vbi.simulator.backend.numpy_.simulator import build_dfun, _build_params
    from vbi.simulator.spec import SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec

    n_nodes = 4
    W = np.zeros((n_nodes, n_nodes))
    spec = SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=0.01),
        coupling=CouplingSpec("linear", a=0.0),
        monitors=(MonitorSpec("raw"),),
        connectivity=Connectivity(weights=W),
    )
    dfun = build_dfun(model)
    params = _build_params(spec)

    state = np.zeros((model.n_sv, n_nodes))
    for i, sv in enumerate(model.state_variables):
        state[i, :] = sv.default_init

    coupling_arr = np.zeros((len(model.cvar), n_nodes))
    deriv = dfun(state, coupling_arr, params)

    assert deriv.shape == state.shape, \
        f"{model.name}: dfun returned wrong shape {deriv.shape}"
    assert np.isfinite(deriv).all(), \
        f"{model.name}: dfun at default_init produced non-finite derivatives"


@pytest.mark.parametrize("model", ALL_NEW_MODELS, ids=lambda m: m.name)
def test_dfun_evaluates_finite_at_zero(model):
    """dfun at all-zeros state with zero coupling must return finite values."""
    from vbi.simulator.backend.numpy_.simulator import build_dfun, _build_params
    from vbi.simulator.spec import SimulationSpec, IntegratorSpec, CouplingSpec, MonitorSpec

    n_nodes = 4
    spec = SimulationSpec(
        model=model,
        integrator=IntegratorSpec(method="heun", dt=0.01),
        coupling=CouplingSpec("linear", a=0.0),
        monitors=(MonitorSpec("raw"),),
        connectivity=Connectivity(weights=np.zeros((n_nodes, n_nodes))),
    )
    dfun = build_dfun(model)
    params = _build_params(spec)
    state = np.zeros((model.n_sv, n_nodes))
    coupling_arr = np.zeros((len(model.cvar), n_nodes))
    deriv = dfun(state, coupling_arr, params)
    assert np.isfinite(deriv).all(), \
        f"{model.name}: dfun at zero state produced non-finite derivatives"
