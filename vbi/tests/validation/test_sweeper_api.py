import sys
from types import SimpleNamespace

import pytest

import vbi.simulator.api as api


class DummySweeper:
    calls = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        DummySweeper.calls.append((args, kwargs))

    def run(self, duration):
        return {"duration": duration}

    def run_df(self, duration):
        return {"duration": duration}


@pytest.fixture(autouse=True)
def clear_dummy_calls():
    DummySweeper.calls.clear()


def test_numba_sweeper_n_workers_sets_numba_threads(monkeypatch):
    calls = []

    monkeypatch.setattr(api, "load_sweep_backend", lambda backend: DummySweeper)
    monkeypatch.setitem(
        sys.modules,
        "numba",
        SimpleNamespace(set_num_threads=lambda n: calls.append(n)),
    )

    api.Sweeper("spec", "sweep", backend="numba", n_workers=4)

    assert calls == [4]
    assert DummySweeper.calls == [(("spec", "sweep"), {})]


def test_cpp_sweeper_n_workers_is_forwarded(monkeypatch):
    monkeypatch.setattr(api, "load_sweep_backend", lambda backend: DummySweeper)

    api.Sweeper("spec", "sweep", backend="cpp", n_workers=4)

    assert DummySweeper.calls == [(("spec", "sweep"), {"n_workers": 4})]


def test_n_workers_rejects_non_positive_values():
    with pytest.raises(ValueError, match="n_workers"):
        api.Sweeper("spec", "sweep", backend="numba", n_workers=0)
