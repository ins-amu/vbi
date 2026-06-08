"""
Tests for VBIInference.from_config - Step 5 of MI6.
"""
import json
import numpy as np
import pytest
from vbi.inference import VBIInference, BoxUniform
from vbi.tests.validation.conftest import make_mpr_spec, make_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_NODES  = 4
DURATION = 200.0
N_SIM    = 10


def _write_connectivity(tmp_path):
    W, D = make_weights(N_NODES, seed=0)
    p = tmp_path / "sc.npz"
    np.savez(p, weights=W, tract_lengths=D)
    return str(p)


def _minimal_config(conn_path: str) -> dict:
    return {
        "sim": {
            "model": "mpr",
            "connectivity": conn_path,
            "dt": 0.01,
            "method": "heun",
            "monitors": [{"kind": "tavg", "period": 1.0}],
            "coupling": {"kind": "linear", "a": 1.0},
            "speed": 4.0,
        },
        "prior": {
            "type": "BoxUniform",
            "low":  [0.1, -6.0],
            "high": [2.0, -3.0],
            "param_names": ["G", "eta"],
        },
        "pipeline": {
            "features": ["calc_mean", "calc_std"],
            "signal":   "tavg",
            "t_cut":    0.0,
        },
        "inference": {
            "density_estimator": "maf",
            "integrator_backend": "numpy",
            "estimator_backend":  "numpy",
            "training": {
                "training_batch_size": 64,
                "stop_after_epochs":   3,
                "max_num_epochs":      5,
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFromConfigDict:

    def test_from_dict_creates_inference_object(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        inf = VBIInference.from_config(_minimal_config(conn))
        assert isinstance(inf, VBIInference)

    def test_prior_param_names(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        inf = VBIInference.from_config(_minimal_config(conn))
        assert inf._prior._resolved_param_names == ["G", "eta"]

    def test_default_train_kwargs_loaded(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        inf = VBIInference.from_config(_minimal_config(conn))
        assert inf._default_train_kwargs["training_batch_size"] == 64
        assert inf._default_train_kwargs["stop_after_epochs"] == 3

    def test_simulate_after_from_config(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        inf = VBIInference.from_config(_minimal_config(conn))
        theta, x = inf.simulate(N_SIM, DURATION, seed=0)
        assert theta.shape == (N_SIM, 2)
        assert x.ndim == 2 and x.shape[0] == N_SIM

    def test_param_names_stored_after_simulate(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        inf = VBIInference.from_config(_minimal_config(conn))
        inf.simulate(N_SIM, DURATION, seed=1)
        assert inf._param_names == ["G", "eta"]

    def test_train_uses_default_kwargs(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        inf = VBIInference.from_config(_minimal_config(conn))
        inf.simulate(N_SIM, DURATION, seed=2)
        est = inf.train()           # uses default stop_after_epochs=3
        assert est is not None

    def test_train_kwargs_overridable(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        inf = VBIInference.from_config(_minimal_config(conn))
        inf.simulate(N_SIM, DURATION, seed=3)
        est = inf.train(stop_after_epochs=2, max_num_epochs=3)
        assert est is not None

    def test_full_round_trip(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        inf = VBIInference.from_config(_minimal_config(conn))
        inf.simulate(N_SIM, DURATION, seed=4)
        est = inf.train()
        post = inf.build_posterior(est)
        x_obs = inf._snpe.get_simulations()[1][0]
        samples = post.sample((30,), x=x_obs)
        assert samples.shape == (30, 2)

    def test_integrator_backend_set(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        inf = VBIInference.from_config(_minimal_config(conn))
        assert inf._integrator_backend == "numpy"

    def test_de_type_set(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        inf = VBIInference.from_config(_minimal_config(conn))
        assert inf._de_type == "maf"


class TestFromConfigFile:

    def test_from_json_file(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        cfg = _minimal_config(conn)
        json_path = tmp_path / "config.json"
        json_path.write_text(json.dumps(cfg))

        inf = VBIInference.from_config(str(json_path))
        assert isinstance(inf, VBIInference)
        theta, x = inf.simulate(N_SIM, DURATION, seed=5)
        assert theta.shape[0] == N_SIM

    def test_from_yaml_file(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        conn = _write_connectivity(tmp_path)
        cfg = _minimal_config(conn)
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(cfg))

        inf = VBIInference.from_config(str(yaml_path))
        assert isinstance(inf, VBIInference)
        theta, x = inf.simulate(N_SIM, DURATION, seed=6)
        assert theta.shape[0] == N_SIM

    def test_unknown_model_raises(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        cfg = _minimal_config(conn)
        cfg["sim"]["model"] = "nonexistent_model_xyz"
        with pytest.raises(ValueError, match="Unknown model"):
            VBIInference.from_config(cfg)

    def test_unknown_prior_type_raises(self, tmp_path):
        conn = _write_connectivity(tmp_path)
        cfg = _minimal_config(conn)
        cfg["prior"]["type"] = "Dirichlet"
        with pytest.raises(ValueError, match="Unknown prior type"):
            VBIInference.from_config(cfg)
