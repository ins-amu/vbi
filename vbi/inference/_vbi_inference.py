"""
VBIInference - end-to-end SBI workflow object.  MI6.

Owns the full loop: prior sampling → sweep simulation → feature extraction
→ SNPE training → posterior.  Raw recordings can optionally be cached to
disk so features can be recomputed with different pipeline settings later.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._api import SNPE
from ._utils import (
    simulate_for_vbi_sweep,
    simulate_for_vbi_sweep_cached,
    extract_from_cache as _extract_from_cache,
)

if TYPE_CHECKING:
    from ._posterior import Posterior
    from ._estimators import ConditionalDensityEstimator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estimator serialisation helpers
# ---------------------------------------------------------------------------

def _pack_estimator(est, data: dict) -> None:
    """
    Write all state needed to reconstruct ``est`` for inference (not training)
    into the flat numpy array dict ``data`` using an ``est__`` prefix.

    Saved:
      - full constructor config (all dataclass fields) as a JSON string
      - weights dict
      - z-score normalizers  (theta_mean/std, x_mean/std)
      - autoregressive masks (model_constants per flow)
      - ActNorm init flags
      - PCA components (if used)
    """
    import dataclasses, json

    # Full constructor config - covers MAF/NSF/MDN non-default architectures.
    if dataclasses.is_dataclass(est):
        config: dict = {}
        for f in dataclasses.fields(est):
            val = getattr(est, f.name, None)
            if isinstance(val, (int, float, bool, str, type(None))):
                config[f.name] = val
            elif isinstance(val, (list, tuple)) and all(isinstance(x, (int, float)) for x in val):
                config[f.name] = list(val)
        data["est__config"] = np.array(json.dumps(config))

    data["est__type"] = np.array(type(est).__name__)

    for k, v in est.weights.items():
        data[f"est__w__{k}"] = np.asarray(v)

    data["est__param_dim"]   = np.array(est.param_dim)
    data["est__feature_dim"] = np.array(est.feature_dim)

    for attr in ("theta_mean", "theta_std", "x_mean", "x_std"):
        val = getattr(est, attr, None)
        if val is not None:
            data[f"est__{attr}"] = np.asarray(val)

    mc = getattr(est, "model_constants", None)
    if mc is not None:
        for k_flow, layer in enumerate(mc["layers"]):
            for key in ("M1", "M2", "Mm", "perm", "inv_perm"):
                if key in layer:
                    data[f"est__mc__{k_flow}__{key}"] = np.asarray(layer[key])

    actnorm = getattr(est, "_actnorm_initialized", None)
    if actnorm is not None:
        data["est__actnorm_init"] = np.array(actnorm, dtype=bool)

    if getattr(est, "_use_pca", False) and getattr(est, "_pca_components", None) is not None:
        data["est__pca_components"] = np.asarray(est._pca_components)


def _unpack_estimator(d, EstCls):
    """
    Reconstruct a fully functional estimator from the flat array dict ``d``
    (keys produced by ``_pack_estimator``).

    Reconstructs from the ``est__config`` JSON blob (full constructor kwargs),
    falling back to legacy field-by-field keys for old checkpoints.
    After construction, restores weights and all runtime state, then rebuilds
    any derived attributes (e.g. MDN ``_offdiag_basis``).
    """
    import dataclasses, json

    _fields = {f.name for f in dataclasses.fields(EstCls)} if dataclasses.is_dataclass(EstCls) else set()

    if "est__config" in d:
        raw = json.loads(str(d["est__config"]))
        # Coerce list → tuple for tuple-typed fields (e.g. hidden_sizes)
        kwargs = {}
        for f in dataclasses.fields(EstCls) if dataclasses.is_dataclass(EstCls) else []:
            if f.name in raw:
                val = raw[f.name]
                # f.default is a tuple (e.g. hidden_sizes=(32,32)) → coerce JSON list
                if isinstance(val, list) and isinstance(f.default, tuple):
                    val = tuple(val)
                kwargs[f.name] = val
        est = EstCls(**{k: v for k, v in kwargs.items() if k in _fields})
    elif "n_flows" in _fields:
        # Legacy checkpoint: MAF/NSF only stored n_flows
        n_flows = int(d["est__n_flows"]) if "est__n_flows" in d else 4
        est = EstCls(n_flows=n_flows)
    else:
        # Legacy checkpoint: MDN stored n_components / hidden_sizes separately
        kwargs = {}
        if "est__n_components" in d:
            kwargs["n_components"] = int(d["est__n_components"])
        if "est__hidden_sizes" in d:
            kwargs["hidden_sizes"] = tuple(int(x) for x in d["est__hidden_sizes"])
        est = EstCls(**kwargs)

    est.weights        = {k[len("est__w__"):]: d[k] for k in d if k.startswith("est__w__")}
    est.param_dim      = int(d["est__param_dim"])
    est.feature_dim    = int(d["est__feature_dim"])
    est._dims_inferred = True
    est.loss_history   = []

    # Rebuild derived state that depends on param_dim/feature_dim.
    # MDNEstimator creates _offdiag_basis in _infer_dimensions(); rebuild it here.
    if hasattr(est, "_create_offdiag_basis"):
        est._offdiag_basis = est._create_offdiag_basis()

    for attr in ("theta_mean", "theta_std", "x_mean", "x_std"):
        if f"est__{attr}" in d:
            setattr(est, attr, d[f"est__{attr}"])

    mc_keys = [k for k in d if k.startswith("est__mc__")]
    if mc_keys:
        n_mc_flows = max(int(k.split("__")[2]) for k in mc_keys) + 1
        layers = []
        for k_flow in range(n_mc_flows):
            layer = {
                key: d[f"est__mc__{k_flow}__{key}"]
                for key in ("M1", "M2", "Mm", "perm", "inv_perm")
                if f"est__mc__{k_flow}__{key}" in d
            }
            layers.append(layer)
        est.model_constants = {"layers": layers}

    if "est__actnorm_init" in d:
        est._actnorm_initialized = list(d["est__actnorm_init"])

    if "est__pca_components" in d:
        est._pca_components = d["est__pca_components"]
        est._use_pca = True

    return est


# ---------------------------------------------------------------------------
# Config-loading helpers  (Step 5)
# ---------------------------------------------------------------------------

def _load_config(config) -> dict:
    """Accept a file path (YAML/JSON) or a plain dict."""
    if isinstance(config, dict):
        return config
    path = Path(config)
    text = path.read_text()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        import json
        return json.loads(text)


def _parse_prior(d: dict):
    """Build a prior object from a config dict with a ``type`` key."""
    from ._prior import (
        BoxUniform, Gaussian, MultivariateNormal,
        LogNormal, Gamma, Beta, MultipleIndependent,
    )
    t = d["type"]
    pn = d.get("param_names")
    if t == "BoxUniform":
        return BoxUniform(
            low=np.asarray(d["low"]),
            high=np.asarray(d["high"]),
            param_names=pn,
        )
    if t == "Gaussian":
        return Gaussian(
            mean=np.asarray(d["mean"]),
            std=np.asarray(d["std"]),
            param_names=pn,
        )
    if t == "MultivariateNormal":
        return MultivariateNormal(
            mean=np.asarray(d["mean"]),
            cov=np.asarray(d["cov"]),
            param_names=pn,
        )
    if t == "LogNormal":
        return LogNormal(np.asarray(d["mean"]), np.asarray(d["std"]), param_names=pn)
    if t == "Gamma":
        return Gamma(np.asarray(d["concentration"]), np.asarray(d["rate"]), param_names=pn)
    if t == "Beta":
        return Beta(np.asarray(d["alpha"]), np.asarray(d["beta"]), param_names=pn)
    if t == "MultipleIndependent":
        return MultipleIndependent(
            [_parse_prior(p) for p in d["priors"]], param_names=pn
        )
    raise ValueError(
        f"Unknown prior type {t!r}. "
        f"Supported: BoxUniform, Gaussian, MultivariateNormal, LogNormal, "
        f"Gamma, Beta, MultipleIndependent."
    )


def _parse_pipeline(d: dict):
    """Build a FeaturePipeline from a config dict."""
    from vbi.feature_extraction import (
        FeaturePipeline,
        get_features_by_domain,
        get_features_by_given_names,
    )
    domain   = d.get("domain")
    features = d.get("features")

    cfg = get_features_by_domain(domain)     # domain=None → all domains
    if features:
        cfg = get_features_by_given_names(cfg, names=list(features))

    return FeaturePipeline(
        cfg,
        signal = d.get("signal", "tavg"),
        t_cut  = float(d.get("t_cut", 500.0)),
    )


def _pack_pruner(pruner, data: dict) -> None:
    """Persist fitted FeaturePruner state into a checkpoint data dict."""
    if pruner is None or getattr(pruner, "kept_mask_", None) is None:
        return

    import json

    data["pruner__min_std"] = np.array(float(pruner.min_std))
    data["pruner__max_corr"] = np.array(float(pruner.max_corr))
    data["pruner__remove_nan"] = np.array(bool(pruner.remove_nan))
    data["pruner__kept_mask"] = np.asarray(pruner.kept_mask_, dtype=bool)
    data["pruner__kept_labels"] = np.array("|".join(pruner.kept_labels_ or []))
    data["pruner__n_original"] = np.array(
        -1 if pruner._n_original is None else int(pruner._n_original)
    )
    data["pruner__removed_reason"] = np.array(
        json.dumps(getattr(pruner, "_removed_reason", {}))
    )


def _unpack_pruner(d, pipeline) -> None:
    """Restore fitted FeaturePruner state onto ``pipeline.pruner`` if present."""
    if "pruner__kept_mask" not in d:
        return

    import json
    from vbi.feature_extraction import FeaturePruner

    pruner = getattr(pipeline, "pruner", None)
    if pruner is None:
        pruner = FeaturePruner(
            min_std=float(d["pruner__min_std"]),
            max_corr=float(d["pruner__max_corr"]),
            remove_nan=bool(d["pruner__remove_nan"]),
        )
        pipeline.pruner = pruner

    pruner.min_std = float(d["pruner__min_std"])
    pruner.max_corr = float(d["pruner__max_corr"])
    pruner.remove_nan = bool(d["pruner__remove_nan"])
    pruner.kept_mask_ = np.asarray(d["pruner__kept_mask"], dtype=bool)

    labels_raw = str(d["pruner__kept_labels"])
    pruner.kept_labels_ = labels_raw.split("|") if labels_raw else []

    n_original = int(d["pruner__n_original"])
    pruner._n_original = None if n_original < 0 else n_original
    if "pruner__removed_reason" in d:
        pruner._removed_reason = json.loads(str(d["pruner__removed_reason"]))
    else:
        pruner._removed_reason = {}


# ---------------------------------------------------------------------------
# SBC simulator helper  (Step 6)
# ---------------------------------------------------------------------------

def _make_simulator_fn(sim_spec, prior, pipeline, integrator_backend, duration):
    """
    Return a callable ``theta_1d -> x_1d`` suitable for :func:`run_sbc`.

    Each call patches the SimulationSpec with the given parameter values,
    runs one simulation, and extracts features via the pipeline.
    """
    from vbi.simulator.spec.sweep import SweepSpec
    from vbi.simulator.api import Sweeper

    param_names = prior._resolved_param_names

    def simulator_fn(theta_1d: np.ndarray) -> np.ndarray:
        theta_1d = np.asarray(theta_1d, dtype=np.float64).ravel()
        sweep_spec = SweepSpec(
            params      = theta_1d[None, :],   # (1, d_theta)
            param_names = tuple(param_names),
            pipeline    = pipeline,
        )
        sweeper = Sweeper(sim_spec, sweep_spec, backend=integrator_backend)
        labels, values = sweeper.run(duration)
        n_params = len(param_names)
        x_1d = values[0, n_params:].astype(np.float64)
        return x_1d

    return simulator_fn


# ---------------------------------------------------------------------------
# SBI inference-backend helpers  (Step 8)
# ---------------------------------------------------------------------------

def _prior_to_sbi(prior):
    """Convert a vbi prior to a sbi/torch-compatible prior."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "inference_backend='sbi' requires PyTorch. "
            "Install with:  pip install torch"
        ) from e
    from vbi.inference._prior import BoxUniform as _VBIBoxUniform

    if isinstance(prior, _VBIBoxUniform):
        from sbi.utils import BoxUniform as _SBIBoxUniform
        return _SBIBoxUniform(
            low  = torch.tensor(prior.low,  dtype=torch.float32),
            high = torch.tensor(prior.high, dtype=torch.float32),
        )

    # Generic wrapper: expose torch Distribution interface
    class _TorchWrapper(torch.distributions.Distribution):
        arg_constraints: dict = {}
        has_rsample = False

        def __init__(self, vbi_prior):
            self._p = vbi_prior
            super().__init__(
                batch_shape=torch.Size([]),
                event_shape=torch.Size([vbi_prior.dim]),
                validate_args=False,
            )

        def sample(self, sample_shape=torch.Size()):
            n = sample_shape[0] if len(sample_shape) > 0 else 1
            arr = self._p.sample((n,))
            return torch.tensor(arr, dtype=torch.float32)

        def log_prob(self, theta):
            arr = theta.detach().cpu().numpy()
            lp  = self._p.log_prob(arr)
            return torch.tensor(lp, dtype=torch.float32)

    return _TorchWrapper(prior)


def _setup_sbi_engine(prior, density_estimator: str, inference_engine, show_progress_bars: bool):
    """Initialise (or return) an sbi SNPE engine."""
    try:
        from sbi.inference import SNPE as _SBI_SNPE
    except ImportError as e:
        raise ImportError(
            "inference_backend='sbi' requires the 'sbi' package. "
            "Install with:  pip install sbi"
        ) from e

    if inference_engine is not None:
        return inference_engine

    torch_prior = _prior_to_sbi(prior)
    return _SBI_SNPE(
        prior             = torch_prior,
        density_estimator = density_estimator,
        show_progress_bars = show_progress_bars,
    )


def _filter_sbi_train_kwargs(kwargs: dict) -> dict:
    """
    Keep only kwargs that sbi's SNPE.train() accepts.
    sbi uses the same kwarg names as vbi for the common ones.
    """
    _SBI_TRAIN_KEYS = {
        "training_batch_size", "learning_rate", "validation_fraction",
        "stop_after_epochs", "max_num_epochs", "clip_max_norm",
        "num_atoms", "resume_training", "show_train_summary",
    }
    return {k: v for k, v in kwargs.items() if k in _SBI_TRAIN_KEYS}


class _NumpyPosteriorWrapper:
    """
    Wraps an sbi Posterior so that ``sample`` and ``log_prob`` return numpy
    arrays instead of torch tensors.  The underlying sbi posterior is
    accessible via ``.sbi_posterior``.
    """

    def __init__(self, sbi_posterior):
        self.sbi_posterior = sbi_posterior

    def sample(self, sample_shape, x=None, seed=None):
        import torch
        n = sample_shape[0] if isinstance(sample_shape, tuple) else int(sample_shape)
        x_t = torch.tensor(np.asarray(x), dtype=torch.float32) if x is not None else None
        samples = self.sbi_posterior.sample(
            (n,), x=x_t, show_progress_bars=False
        )
        return samples.detach().cpu().numpy()

    def log_prob(self, theta, x=None):
        import torch
        theta_t = torch.tensor(np.atleast_2d(theta), dtype=torch.float32)
        x_t     = torch.tensor(np.asarray(x), dtype=torch.float32) if x is not None else None
        lp = self.sbi_posterior.log_prob(theta_t, x=x_t)
        return lp.detach().cpu().numpy()

    def set_default_x(self, x):
        import torch
        self.sbi_posterior.set_default_x(
            torch.tensor(np.asarray(x), dtype=torch.float32)
        )
        return self

    def __repr__(self):
        return f"_NumpyPosteriorWrapper({self.sbi_posterior!r})"


class VBIInference:
    """
    End-to-end VBI inference workflow.

    Owns the full loop: prior sampling → sweep simulation → feature
    extraction → SNPE training → posterior.

    Parameters
    ----------
    sim_spec : SimulationSpec
    prior : prior object
        Must expose ``._resolved_param_names``, ``.sample()``, ``.log_prob()``.
    pipeline : FeaturePipeline
    density_estimator : 'maf' | 'mdn' | 'nsf'
    integrator_backend : str
        Backend for the Sweeper ('numpy' | 'numba' | 'cuda' | 'jax').
    estimator_backend : str
        Backend for the density estimator ('auto' | 'numpy' | 'jax').
    show_progress_bars : bool
    embedding_net : EmbeddingNet | None

    Examples
    --------
    >>> inf = VBIInference(sim_spec, prior, pipeline)
    >>> theta, x = inf.simulate(num_simulations=2000, duration=5000.0)
    >>> est = inf.train(training_batch_size=256, stop_after_epochs=30)
    >>> post = inf.build_posterior(est)
    >>> samples = post.sample((1000,), x=x_obs)
    >>> inf.save("run.npz")
    """

    def __init__(
        self,
        sim_spec,
        prior,
        pipeline,
        density_estimator: str = "maf",
        integrator_backend: str = "numba",
        estimator_backend: str = "auto",
        show_progress_bars: bool = True,
        embedding_net=None,
        inference_backend: str = "vbi",
        inference_engine=None,
    ):
        """
        Parameters
        ----------
        inference_backend : 'vbi' | 'sbi'
            Which inference engine to use for training and posterior.
            'vbi' (default) uses the built-in torch-free SNPE.
            'sbi' uses ``sbi.inference.SNPE`` - requires ``sbi`` and ``torch``
            to be installed.
        inference_engine : sbi.inference.SNPE | None
            Pass a pre-configured sbi SNPE object directly.  When set,
            ``inference_backend`` is ignored and the given object is used.
            The prior is taken from ``prior`` (converted to torch).
        """
        self._sim_spec            = sim_spec
        self._prior               = prior
        self._pipeline            = pipeline
        self._integrator_backend  = integrator_backend
        self._de_type             = density_estimator
        self._show_progress_bars  = show_progress_bars
        self._feature_labels: list[str] | None = None
        self._param_names:    list[str] | None = None
        self._default_train_kwargs: dict       = {}
        self._last_estimator                   = None
        self._sim_rounds: list[tuple]          = []   # (theta, x) per simulate() call

        if inference_engine is not None or inference_backend == "sbi":
            self._inference_backend = "sbi"
            self._snpe              = None
            self._sbi_engine        = _setup_sbi_engine(
                prior, density_estimator, inference_engine, show_progress_bars,
            )
        else:
            self._inference_backend = "vbi"
            self._sbi_engine        = None
            self._snpe              = SNPE(
                prior               = prior,
                density_estimator   = density_estimator,
                backend             = estimator_backend,
                show_progress_bars  = show_progress_bars,
                embedding_net       = embedding_net,
            )

    # ------------------------------------------------------------------
    # Core workflow
    # ------------------------------------------------------------------

    def simulate(
        self,
        num_simulations: int,
        duration: float,
        seed: int | None = None,
        proposal=None,
        x_obs=None,
        cache_dir=None,
        chunk_size: int = 500,
        n_workers: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample parameters and run the sweep simulator.

        Appends the resulting ``(theta, x)`` pairs to the internal SNPE
        object, ready for training.

        Parameters
        ----------
        num_simulations : int
        duration : float  simulation length in ms
        seed : int | None
        proposal : Posterior | None
            When set, sample theta from this posterior instead of the prior.
            Requires ``x_obs``.
        x_obs : ndarray | None  required when proposal is not None
        cache_dir : str | Path | None
            When set, raw monitor recordings are written to this directory
            in chunked ``.npz`` files.  Use ``extract_from_cache()`` later
            to re-extract features with a different pipeline.
        chunk_size : int
            Simulations per chunk (only used when cache_dir is set).
            Tune to stay within GPU/RAM budget.
        n_workers : int | None
            Number of threads for the numba backend.  None = use all available
            (``numba.get_num_threads()``).  Also sets the progress batch size
            when ``show_progress_bars=True``.

        Returns
        -------
        theta : (n, d_theta) ndarray
        x     : (n, d_x)     ndarray
        """
        common = dict(
            sim_spec            = self._sim_spec,
            prior               = self._prior,
            pipeline            = self._pipeline,
            num_simulations     = num_simulations,
            duration            = duration,
            sim_backend         = self._integrator_backend,
            seed                = seed,
            proposal            = proposal,
            x_obs               = x_obs,
            show_progress_bars  = self._show_progress_bars,
            n_workers           = n_workers,
        )

        if cache_dir is None:
            theta, x, param_names, feat_labels = simulate_for_vbi_sweep(**common)
        else:
            theta, x, param_names, feat_labels = simulate_for_vbi_sweep_cached(
                **common, cache_dir=cache_dir, chunk_size=chunk_size
            )

        # Apply feature pruning if the pipeline carries a FeaturePruner.
        # Round 1: fit the mask on this sweep's x, then transform.
        # Round 2+: reuse the already-fitted mask (same feature selection).
        pruner = getattr(self._pipeline, "pruner", None)
        if pruner is not None:
            if pruner.kept_mask_ is None:
                x, feat_labels = pruner.fit_transform(x, list(feat_labels))
            else:
                x = pruner.transform(x)
                feat_labels = list(pruner.kept_labels_)

        if self._param_names is None:
            self._param_names = param_names
        if self._feature_labels is None:
            self._feature_labels = feat_labels

        # Backend-agnostic storage (used by get_simulations / save / load)
        self._sim_rounds.append((theta.copy(), x.copy()))

        # Append to the active inference engine
        if self._inference_backend == "vbi":
            self._snpe.append_simulations(theta, x, proposal=proposal)
        else:
            import torch
            theta_t = torch.tensor(theta, dtype=torch.float32)
            x_t     = torch.tensor(x,     dtype=torch.float32)
            if proposal is not None:
                self._sbi_engine.append_simulations(theta_t, x_t, proposal=proposal)
            else:
                self._sbi_engine.append_simulations(theta_t, x_t)

        return theta, x

    def train(self, **train_kwargs):
        """
        Train the density estimator on all accumulated simulations.

        Keyword arguments are forwarded to ``SNPE.train()`` (vbi backend) or
        ``sbi.SNPE.train()`` (sbi backend).  Any kwargs stored via
        ``from_config`` serve as defaults and can be overridden here.

        Returns
        -------
        estimator  (vbi backend: ConditionalDensityEstimator;
                    sbi backend: sbi NeuralPosteriorEstimator)
        """
        if not self._sim_rounds:
            raise RuntimeError(
                "No simulations available.  Call simulate() before train()."
            )
        merged = {**self._default_train_kwargs, **train_kwargs}
        # Propagate show_progress_bars as verbose unless caller set it explicitly
        if "verbose" not in merged:
            merged["verbose"] = self._show_progress_bars

        if self._inference_backend == "vbi":
            self._last_estimator = self._snpe.train(**merged)
        else:
            # sbi kwarg names overlap substantially with vbi's; pass all through
            self._last_estimator = self._sbi_engine.train(**_filter_sbi_train_kwargs(merged))

        return self._last_estimator

    def build_posterior(
        self,
        estimator=None,
        sample_with: str = "direct",
        **kwargs,
    ):
        """
        Wrap the trained estimator in a Posterior object.

        For the vbi backend returns a ``vbi.inference.Posterior``.
        For the sbi backend returns a numpy-compatible wrapper around the
        sbi posterior (``sample`` and ``log_prob`` return numpy arrays).

        Parameters
        ----------
        estimator : estimator object | None
            Uses the last one returned by ``train()`` if None.
        sample_with : 'direct' | 'mcmc' | 'rejection'
            Only used by the vbi backend.
        """
        est = estimator if estimator is not None else self._last_estimator
        if self._inference_backend == "vbi":
            return self._snpe.build_posterior(
                density_estimator=est, sample_with=sample_with, **kwargs
            )
        else:
            sbi_posterior = self._sbi_engine.build_posterior(est, **kwargs)
            return _NumpyPosteriorWrapper(sbi_posterior)

    def get_simulations(self, starting_round: int = 0):
        """
        Return all accumulated ``(theta, x)`` from ``starting_round`` onward.

        Works for both vbi and sbi backends.

        Returns
        -------
        theta     : (N, d_theta) float32
        x         : (N, d_x)    float32
        proposals : list        (always empty list for sbi backend)
        """
        rounds = self._sim_rounds[starting_round:]
        if not rounds:
            empty = np.empty((0, 0), dtype=np.float32)
            return empty, empty, []
        theta = np.concatenate([r[0] for r in rounds], axis=0).astype(np.float32)
        x     = np.concatenate([r[1] for r in rounds], axis=0).astype(np.float32)
        return theta, x, []

    def append_simulations(
        self,
        theta,
        x,
        *,
        param_names: list[str] | tuple[str, ...] | None = None,
        feature_labels: list[str] | tuple[str, ...] | None = None,
        proposal=None,
    ) -> "VBIInference":
        """
        Append precomputed ``(theta, x)`` pairs to the active inference engine.

        This is useful when features are re-extracted from cached raw recordings
        rather than produced by :meth:`simulate`.
        """
        theta = np.asarray(theta, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if theta.ndim != 2 or x.ndim != 2:
            raise ValueError(f"theta and x must be 2-D, got {theta.shape}, {x.shape}.")
        if theta.shape[0] != x.shape[0]:
            raise ValueError(
                f"theta and x must have the same row count, got "
                f"{theta.shape[0]} and {x.shape[0]}."
            )

        if param_names is not None:
            self._param_names = list(param_names)
        elif self._param_names is None:
            self._param_names = list(getattr(self._prior, "_resolved_param_names", []))

        if feature_labels is not None:
            self._feature_labels = list(feature_labels)

        self._sim_rounds.append((theta.copy(), x.copy()))

        if self._inference_backend == "vbi":
            self._snpe.append_simulations(theta, x, proposal=proposal)
        else:
            import torch

            theta_t = torch.tensor(theta, dtype=torch.float32)
            x_t = torch.tensor(x, dtype=torch.float32)
            if proposal is not None:
                self._sbi_engine.append_simulations(theta_t, x_t, proposal=proposal)
            else:
                self._sbi_engine.append_simulations(theta_t, x_t)

        return self

    # ------------------------------------------------------------------
    # Cache helper (static - usable without an instance)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_from_cache(cache_dir, pipeline) -> tuple[np.ndarray, np.ndarray]:
        """
        Load cached raw recordings and extract features with a new pipeline.

        Parameters
        ----------
        cache_dir : str | Path
        pipeline  : FeaturePipeline

        Returns
        -------
        theta : (n, d_theta) float64
        x     : (n, d_x)     float64
        """
        return _extract_from_cache(cache_dir, pipeline)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """
        Persist inference state to a single ``.npz`` checkpoint.

        Saved: simulation data (theta, x across all rounds), full
        estimator state (weights + normalizers + masks), feature/param
        label lists, and constructor kwargs.

        For the sbi backend, the trained estimator is written to a companion
        ``<stem>_sbi.pt`` file next to the ``.npz``.

        NOT saved: ``sim_spec`` and ``pipeline`` - supply them on load.

        Parameters
        ----------
        path : str | Path  target file (``.npz`` extension added if absent)
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npz")

        data: dict[str, np.ndarray] = {}

        # -- simulation data (backend-agnostic) ------------------------------
        if self._sim_rounds:
            theta_all, x_all, _ = self.get_simulations()
            data["sim__theta_all"] = theta_all
            data["sim__x_all"]     = x_all

        # -- estimator state (vbi backend only) --------------------------------
        if self._inference_backend == "vbi" and self._last_estimator is not None:
            _pack_estimator(self._last_estimator, data)
        elif self._inference_backend == "sbi" and self._last_estimator is not None:
            import torch
            sbi_path = path.with_name(path.stem + "_sbi.pt")
            torch.save(self._last_estimator, sbi_path)
            log.info("VBIInference sbi estimator saved to %s", sbi_path)

        _pack_pruner(getattr(self._pipeline, "pruner", None), data)

        # -- metadata --------------------------------------------------------
        data["meta__feature_labels"]    = np.array("|".join(self._feature_labels or []))
        data["meta__param_names"]       = np.array("|".join(self._param_names or []))
        data["meta__density_estimator"] = np.array(self._de_type)
        data["meta__integrator_backend"] = np.array(self._integrator_backend)
        data["meta__inference_backend"]  = np.array(self._inference_backend)
        data["meta__n_rounds"]          = np.array(len(self._sim_rounds))
        # VBI-only: persist the resolved estimator backend string
        vbi_backend = self._snpe._backend if self._snpe is not None else "numpy"
        data["meta__backend"]           = np.array(vbi_backend)

        np.savez(path, **data)
        log.info("VBIInference saved to %s", path)

    @classmethod
    def load(
        cls,
        path,
        sim_spec,
        pipeline,
        prior=None,
        show_progress_bars: bool = True,
        embedding_net=None,
    ) -> "VBIInference":
        """
        Restore from a checkpoint written by ``save()``.

        Parameters
        ----------
        path : str | Path
        sim_spec : SimulationSpec
        pipeline : FeaturePipeline  may differ from the saved one
        prior : prior object | None
            Required to call ``build_posterior()`` after load.

        Returns
        -------
        VBIInference  fully reconstructed - ``train()`` and
                      ``build_posterior()`` work immediately.
        """
        from ._backends import resolve_backend, get_estimator_map

        path = Path(path)
        d    = np.load(path, allow_pickle=False)

        de_type            = str(d["meta__density_estimator"])
        integrator_backend = str(d["meta__integrator_backend"]) \
                             if "meta__integrator_backend" in d \
                             else str(d["meta__sim_backend"]) \
                             if "meta__sim_backend" in d else "numba"
        estimator_backend  = str(d["meta__backend"])
        inference_backend  = str(d["meta__inference_backend"]) \
                             if "meta__inference_backend" in d else "vbi"

        inf = cls(
            sim_spec           = sim_spec,
            prior              = prior,
            pipeline           = pipeline,
            density_estimator  = de_type,
            integrator_backend = integrator_backend,
            estimator_backend  = estimator_backend,
            show_progress_bars = show_progress_bars,
            embedding_net      = embedding_net,
            inference_backend  = inference_backend,
        )

        fl_raw = str(d["meta__feature_labels"])
        pn_raw = str(d["meta__param_names"])
        inf._feature_labels = fl_raw.split("|") if fl_raw else []
        inf._param_names    = pn_raw.split("|") if pn_raw else []

        _unpack_pruner(d, inf._pipeline)

        if "sim__theta_all" in d and "sim__x_all" in d:
            theta_all = np.asarray(d["sim__theta_all"])
            x_all     = np.asarray(d["sim__x_all"])
            # Re-inject into the active engine
            if inf._inference_backend == "vbi":
                inf._snpe.append_simulations(theta_all, x_all, exclude_invalid_x=False)
            else:
                import torch
                inf._sbi_engine.append_simulations(
                    torch.tensor(theta_all, dtype=torch.float32),
                    torch.tensor(x_all,     dtype=torch.float32),
                )
            # Restore backend-agnostic round storage (single round)
            inf._sim_rounds = [(theta_all.astype(np.float64), x_all.astype(np.float64))]

        if "est__param_dim" in d and inf._inference_backend == "vbi":
            backend_resolved = resolve_backend(estimator_backend)
            de_map           = get_estimator_map(backend_resolved)
            EstCls           = de_map[de_type]
            est = _unpack_estimator(d, EstCls)
            inf._last_estimator  = est
            inf._snpe._estimator = est
        elif inf._inference_backend == "sbi":
            sbi_path = path.with_name(path.stem + "_sbi.pt")
            if sbi_path.exists():
                import torch
                inf._last_estimator = torch.load(sbi_path, weights_only=False)
                log.info("VBIInference sbi estimator loaded from %s", sbi_path)

        n_sims = sum(r[0].shape[0] for r in inf._sim_rounds)
        log.info(
            "VBIInference loaded from %s  (n_sims=%d, estimator=%s)",
            path, n_sims,
            "restored" if inf._last_estimator is not None else "not found",
        )
        return inf

    # ------------------------------------------------------------------
    # Config loader  (Step 5)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config) -> "VBIInference":
        """
        Build a ``VBIInference`` from a YAML/JSON config file or dict.

        Parameters
        ----------
        config : str | Path | dict
            Path to a YAML or JSON file, or an already-parsed dict.

        Config schema
        -------------
        .. code-block:: yaml

            sim:
              model: mpr
              connectivity: data/SC_68.npz   # .npz with weights / tract_lengths
              node_params: {eta: -4.6}
              dt: 0.1
              method: heun
              monitors: [{kind: tavg, period: 1.0}]
              coupling: {kind: linear, a: 1.0}
              speed: 4.0

            prior:
              type: BoxUniform
              low:  [0.5, -5.5]
              high: [5.0, -3.0]
              param_names: [G, eta]

            pipeline:
              features: [calc_fc, calc_fcd]   # or domain: connectivity
              signal: tavg
              t_cut: 500.0

            inference:
              density_estimator: maf
              inference_backend: vbi        # 'vbi' (default) or 'sbi'
              integrator_backend: numba
              estimator_backend: auto
              training:              # stored as default_train_kwargs
                training_batch_size: 256
                stop_after_epochs: 30
                learning_rate: 5.0e-4
        """
        cfg = _load_config(config)

        from vbi.simulator.spec.simulation import SimulationSpec
        sim_spec = SimulationSpec.from_dict(cfg["sim"])

        prior  = _parse_prior(cfg["prior"])
        pipeline = _parse_pipeline(cfg["pipeline"])

        infer_cfg = cfg.get("inference", {})
        inf = cls(
            sim_spec          = sim_spec,
            prior             = prior,
            pipeline          = pipeline,
            density_estimator = infer_cfg.get("density_estimator", "maf"),
            inference_backend  = infer_cfg.get("inference_backend", "vbi"),
            integrator_backend = infer_cfg.get("integrator_backend",
                                               infer_cfg.get("sim_backend", "numba")),
            estimator_backend  = infer_cfg.get("estimator_backend",
                                               infer_cfg.get("backend", "auto")),
        )
        inf._default_train_kwargs = dict(infer_cfg.get("training", {}))
        return inf

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def plot_loss(self):
        """
        Plot training and validation loss curves for the last ``train()`` call.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        RuntimeError  if ``train()`` has not been called yet.
        """
        from ._diagnostics import plot_loss as _plot_loss

        if self._last_estimator is None:
            raise RuntimeError("No trained estimator - call train() first.")

        train_loss = getattr(self._last_estimator, "loss_history", [])
        val_loss   = getattr(self._last_estimator, "val_loss_history", None)
        if not train_loss:
            raise RuntimeError(
                "Loss history is empty - the estimator may have been loaded "
                "from a checkpoint (loss is not persisted)."
            )
        return _plot_loss(train_loss, val_loss)

    def pairplot(self, x_obs, num_samples: int = 1000, **kwargs):
        """
        Sample the posterior at ``x_obs`` and plot a pairplot.

        Parameters
        ----------
        x_obs       : array_like  observed feature vector
        num_samples : int  posterior samples to draw
        **kwargs    : forwarded to :func:`vbi.inference.pairplot`

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        RuntimeError  if ``build_posterior()`` has not been called.
        """
        from ._diagnostics import pairplot as _pairplot

        if self._last_estimator is None:
            raise RuntimeError(
                "No trained estimator - call train() first."
            )
        posterior = self.build_posterior()
        samples   = posterior.sample((num_samples,), x=np.asarray(x_obs))
        return _pairplot(
            samples,
            labels = self._param_names,
            **kwargs,
        )

    def run_sbc(
        self,
        duration: float | None = None,
        num_sbc_runs: int       = 500,
        num_posterior_samples: int = 100,
        seed: int               = 0,
    ) -> dict:
        """
        Run Simulation-Based Calibration using the internal simulator.

        Parameters
        ----------
        duration              : float  simulation length in ms; uses the last
                                ``simulate()`` duration if None (not stored -
                                must supply explicitly).
        num_sbc_runs          : int
        num_posterior_samples : int
        seed                  : int

        Returns
        -------
        dict  with keys ``ranks``, ``dap_samples``; pass to
              :func:`vbi.inference.check_sbc` or :func:`vbi.inference.sbc_rank_plot`.

        Raises
        ------
        RuntimeError  if ``train()`` / ``build_posterior()`` not called yet, or
                      if ``duration`` is not supplied.
        """
        from ._diagnostics import run_sbc as _run_sbc

        if duration is None:
            raise RuntimeError(
                "duration must be supplied (simulation length in ms)."
            )
        if self._last_estimator is None:
            raise RuntimeError(
                "No trained estimator - call train() first."
            )

        posterior    = self.build_posterior()
        simulator_fn = _make_simulator_fn(
            self._sim_spec, self._prior, self._pipeline,
            self._integrator_backend, duration,
        )
        return _run_sbc(
            posterior            = posterior,
            simulator            = simulator_fn,
            prior                = self._prior,
            num_sbc_runs         = num_sbc_runs,
            num_posterior_samples = num_posterior_samples,
            seed                 = seed,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_sims = sum(r[0].shape[0] for r in self._sim_rounds)
        return (
            f"VBIInference("
            f"density_estimator={self._de_type!r}, "
            f"inference_backend={self._inference_backend!r}, "
            f"integrator_backend={self._integrator_backend!r}, "
            f"n_rounds={len(self._sim_rounds)}, "
            f"n_sims={n_sims})"
        )
