# ---------------------------
# Pairplot (no torch required)
# ---------------------------

import copy
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase
from scipy.stats import gaussian_kde

# ---- tiny utils (no torch)

def _ensure_numpy_no_torch(t) -> np.ndarray:
    """Return np.ndarray for lists/arrays; avoids any torch handling."""
    if isinstance(t, np.ndarray):
        return t
    return np.asarray(t)

def _convert_to_list_of_numpy_no_torch(
    arr: Union[List[np.ndarray], np.ndarray, List[List[float]]]
) -> List[np.ndarray]:
    if not isinstance(arr, list):
        return [ _ensure_numpy_no_torch(arr) ]
    return [ _ensure_numpy_no_torch(a) for a in arr ]

def _handle_nan_infs_no_torch(samples: List[np.ndarray]) -> List[np.ndarray]:
    for i in range(len(samples)):
        if np.isinf(samples[i]).any():
            # cast inf to nan, omit next
            np.nan_to_num(samples[i], copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
        samples[i] = samples[i][~np.isnan(samples[i]).any(axis=1)]
    return samples

def _infer_limits_no_torch(
    samples: List[np.ndarray],
    dim: int,
    points: Optional[List[np.ndarray]] = None,
    eps: float = 0.1,
) -> List[List[float]]:
    limits = []
    for d in range(dim):
        mn = min(np.min(s[:, d]) for s in samples)
        mx = max(np.max(s[:, d]) for s in samples)
        if points is not None and len(points) > 0:
            mn = min(mn, min(np.min(p[:, d]) for p in points))
            mx = max(mx, max(np.max(p[:, d]) for p in points))
        span = mx - mn
        pad = eps * span
        limits.append([mn - pad, mx + pad])
    return limits

def _prepare_for_plot_no_torch(
    samples: Union[List[np.ndarray], np.ndarray, List[List[float]]],
    limits: Optional[Union[List, np.ndarray]] = None,
    points: Optional[Union[List[np.ndarray], np.ndarray, List[List[float]]]] = None,
) -> Tuple[List[np.ndarray], int, np.ndarray, List[np.ndarray]]:
    samples = _convert_to_list_of_numpy_no_torch(samples)
    pts_list: List[np.ndarray] = []
    if points is not None:
        pts_list = _convert_to_list_of_numpy_no_torch(points)
    samples = _handle_nan_infs_no_torch(samples)
    dim = samples[0].shape[1]
    if limits is None or limits == []:
        limits = _infer_limits_no_torch(samples, dim, pts_list)
    else:
        limits = [limits[0] for _ in range(dim)] if len(limits) == 1 else limits
    limits_arr = np.asarray(limits, dtype=float)
    return samples, dim, limits_arr, pts_list

# ---- small adapters (no torch)

def _to_list_string(x: Optional[Union[str, List[Optional[str]]]], n: int) -> List[Optional[str]]:
    return [x for _ in range(n)] if not isinstance(x, list) else x

def _to_list_kwargs(x: Optional[Union[Dict, List[Optional[Dict]]]], n: int) -> List[Optional[Dict]]:
    return [x for _ in range(n)] if not isinstance(x, list) else x

def _update_dict(d: Dict, u: Optional[Dict]) -> Dict:
    if u is not None:
        for k, v in u.items():
            dv = d.get(k, {})
            if isinstance(dv, dict) and isinstance(v, dict):
                d[k] = _update_dict(dv, v)
            else:
                d[k] = v
    return d

# ---- default kwargs (reuse your styling)

def _get_default_fig_kwargs_no_torch() -> Dict:
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    return {
        "legend": None,
        "legend_kwargs": {},
        "points_labels": [f"points_{idx}" for idx in range(10)],
        "samples_labels": [f"samples_{idx}" for idx in range(10)],
        "samples_colors": plt.rcParams["axes.prop_cycle"].by_key()["color"][0::2],
        "points_colors":  plt.rcParams["axes.prop_cycle"].by_key()["color"][1::2],
        "tickformatter": mpl.ticker.FormatStrFormatter("%g"),  # type: ignore
        "tick_labels": None,
        "points_diag": {},
        "points_offdiag": {"marker": ".", "markersize": 10},
        "fig_bg_colors": {"offdiag": None, "diag": None, "lower": None},
        "fig_subplots_adjust": {"top": 0.9},
        "subplots": {},
        "despine": {"offset": 5},
        "title": None,
        "title_format": {"fontsize": 16},
        "x_lim_add_eps": 1e-5,
        "square_subplots": True,
    }

def _get_default_diag_kwargs_no_torch(diag: Optional[str], i: int = 0) -> Dict:
    from matplotlib import pyplot as plt
    if diag == "kde":
        return {"bw_method": "scott", "bins": 50,
                "mpl_kwargs": {"color": plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2]}}
    if diag == "hist":
        return {"bin_heuristic": "Freedman-Diaconis",
                "mpl_kwargs": {"color": plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2],
                               "density": False, "histtype": "step"}}
    if diag == "scatter":
        return {"mpl_kwargs": {"color": plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2]}}
    return {}

def _get_default_offdiag_kwargs_no_torch(offdiag: Optional[str], i: int = 0) -> Dict:
    from matplotlib import pyplot as plt
    if offdiag in ("kde", "kde2d"):
        return {"bw_method": "scott", "bins": 50,
                "mpl_kwargs": {"cmap": "viridis", "origin": "lower", "aspect": "auto"}}
    if offdiag in ("hist", "hist2d"):
        return {"bin_heuristic": None,
                "mpl_kwargs": {"cmap": "viridis", "origin": "lower", "aspect": "auto"},
                "np_hist_kwargs": {"bins": 50, "density": False}}
    if offdiag == "scatter":
        return {"mpl_kwargs": {"color": plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2],
                               "edgecolor": "white", "alpha": 0.5, "rasterized": False}}
    if offdiag in ("contour", "contourf"):
        return {"bw_method": "scott", "bins": 50, "levels": [0.68, 0.95, 0.99],
                "percentile": True,
                "mpl_kwargs": {"colors": plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2]}}
    if offdiag == "plot":
        return {"mpl_kwargs": {"color": plt.rcParams["axes.prop_cycle"].by_key()["color"][i * 2],
                               "aspect": "auto"}}
    return {}

# ---- plotting primitives (no torch names anywhere)

def _hist_1d(ax: Axes, samples: np.ndarray, limits_1d: np.ndarray, diag_kwargs: Dict) -> None:
    hk = copy.deepcopy(diag_kwargs.get("mpl_kwargs", {}))
    bins = hk.get("bins", None)
    if bins is None and diag_kwargs.get("bin_heuristic") == "Freedman-Diaconis":
        from scipy.stats import iqr
        binsize = 2 * iqr(samples) * (len(samples) ** (-1/3))
        bins = np.arange(limits_1d[0], limits_1d[1] + binsize, binsize)
    if isinstance(bins, int):
        bins = np.linspace(limits_1d[0], limits_1d[1], bins)
    if bins is not None:
        hk["bins"] = bins
    ax.hist(samples, **hk)

def _kde_1d(ax: Axes, samples: np.ndarray, limits_1d: np.ndarray, diag_kwargs: Dict) -> None:
    density = gaussian_kde(samples, bw_method=diag_kwargs.get("bw_method", "scott"))
    xs = np.linspace(limits_1d[0], limits_1d[1], diag_kwargs.get("bins", 50))
    ys = density(xs)
    ax.plot(xs, ys, **diag_kwargs.get("mpl_kwargs", {}))

def _scatter_1d(ax: Axes, samples: np.ndarray, _limits: np.ndarray, diag_kwargs: Dict) -> None:
    for s in samples:
        ax.axvline(s, **diag_kwargs.get("mpl_kwargs", {}))

def _hist_2d(ax: Axes, x: np.ndarray, y: np.ndarray,
             limx: Union[np.ndarray, List[float]],
             limy: Union[np.ndarray, List[float]], offdiag_kwargs: Dict) -> None:
    hk = copy.deepcopy(offdiag_kwargs)
    H, xedges, yedges = np.histogram2d(
        x, y,
        range=[[float(limx[0]), float(limx[1])], [float(limy[0]), float(limy[1])]],
        **hk.get("np_hist_kwargs", {"bins": 50})
    )
    ax.imshow(
        H.T,
        extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
        **hk.get("mpl_kwargs", {"origin": "lower", "aspect": "auto"})
    )

def _kde_grid(x: np.ndarray, y: np.ndarray,
              limx: np.ndarray, limy: np.ndarray, offdiag_kwargs: Dict):
    density = gaussian_kde(np.vstack([x, y]), bw_method=offdiag_kwargs.get("bw_method", "scott"))
    X, Y = np.meshgrid(
        np.linspace(limx[0], limx[1], offdiag_kwargs.get("bins", 50)),
        np.linspace(limy[0], limy[1], offdiag_kwargs.get("bins", 50)),
    )
    pos = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(density(pos).T, X.shape)
    if offdiag_kwargs.get("percentile") and "levels" in offdiag_kwargs:
        # normalize to [0,1] then produce cumulative contours
        Zn = (Z - Z.min()) / (Z.max() - Z.min() + 1e-12)
        levels = np.asarray(offdiag_kwargs["levels"])
        idx = Zn.ravel().argsort()[::-1]
        cum = np.cumsum(Zn.ravel()[idx])
        cum = cum / (cum[-1] + 1e-12)
        # Map cumulative mass back to grid, then threshold via levels
        inv = np.empty_like(idx)
        inv[idx] = np.arange(idx.size)
        C = cum[inv].reshape(Z.shape)
        # Convert to "which level" label image for contour—return continuous map
        Z = C
    else:
        Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-12)
    return X, Y, Z

def _kde_2d(ax: Axes, x: np.ndarray, y: np.ndarray,
            limx: np.ndarray, limy: np.ndarray, offdiag_kwargs: Dict) -> None:
    X, Y, Z = _kde_grid(x, y, limx, limy, offdiag_kwargs)
    ax.imshow(
        Z,
        extent=(float(limx[0]), float(limx[1]), float(limy[0]), float(limy[1])),
        **offdiag_kwargs.get("mpl_kwargs", {"origin": "lower", "aspect": "auto"})
    )

def _contour_2d(ax: Axes, x: np.ndarray, y: np.ndarray,
                limx: np.ndarray, limy: np.ndarray, offdiag_kwargs: Dict) -> None:
    X, Y, Z = _kde_grid(x, y, limx, limy, offdiag_kwargs)
    ax.contour(
        X, Y, Z,
        levels=offdiag_kwargs.get("levels", [0.68, 0.95, 0.99]),
        **offdiag_kwargs.get("mpl_kwargs", {})
    )

def _scatter_2d(ax: Axes, x: np.ndarray, y: np.ndarray,
                _lx: np.ndarray, _ly: np.ndarray, offdiag_kwargs: Dict) -> None:
    ax.scatter(x, y, **offdiag_kwargs.get("mpl_kwargs", {}))

def _plot_2d(ax: Axes, x: np.ndarray, y: np.ndarray,
             _lx: np.ndarray, _ly: np.ndarray, offdiag_kwargs: Dict) -> None:
    ax.plot(x, y, **offdiag_kwargs.get("mpl_kwargs", {}))

def _get_diag_funcs_no_torch(diag_list: List[Optional[str]]
) -> List[Optional[Callable[[Axes, np.ndarray, np.ndarray, Dict], None]]]:
    out = []
    for d in diag_list:
        out.append({"hist": _hist_1d, "kde": _kde_1d, "scatter": _scatter_1d}.get(d))
    return out

def _get_offdiag_funcs_no_torch(off_list: List[Optional[str]]
) -> List[Optional[Callable[[Axes, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict], None]]]:
    out = []
    for od in off_list:
        out.append({
            "hist": _hist_2d, "hist2d": _hist_2d,
            "kde": _kde_2d, "kde2d": _kde_2d,
            "contour": _contour_2d, "contourf": _contour_2d,
            "scatter": _scatter_2d, "plot": _plot_2d
        }.get(od))
    return out

# ---- subplot formatting (no torch)

def _format_axis(ax: Axes, xhide=True, yhide=True, xlabel="", ylabel="", tickformatter=None) -> Axes:
    for loc in ["right", "top", "left", "bottom"]:
        ax.spines[loc].set_visible(False)
    if xhide:
        ax.set_xlabel("")
        ax.xaxis.set_ticks_position("none")
        ax.xaxis.set_tick_params(labelbottom=False)
    if yhide:
        ax.set_ylabel("")
        ax.yaxis.set_ticks_position("none")
        ax.yaxis.set_tick_params(labelleft=False)
    if not xhide:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_tick_params(labelbottom=True)
        if tickformatter is not None:
            ax.xaxis.set_major_formatter(tickformatter)
        ax.spines["bottom"].set_visible(True)
    if not yhide:
        ax.set_ylabel(ylabel)
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_tick_params(labelleft=True)
        if tickformatter is not None:
            ax.yaxis.set_major_formatter(tickformatter)
        ax.spines["left"].set_visible(True)
    return ax

def _format_subplot_np(
    ax: Axes,
    current: str,
    limits: Union[List[List[float]], np.ndarray],
    ticks: Optional[Union[List, np.ndarray]],
    labels_dim: List[str],
    fig_kwargs: Dict,
    row: int,
    col: int,
    dim: int,
    flat: bool,
    excl_lower: bool,
) -> None:
    if isinstance(limits, np.ndarray):
        lims = limits.tolist()
    else:
        lims = limits

    if current == "diag":
        eps = fig_kwargs["x_lim_add_eps"]
        ax.set_xlim((lims[col][0] - eps, lims[col][1] + eps))
    else:
        ax.set_xlim((lims[col][0], lims[col][1]))
        ax.set_ylim((lims[row][0], lims[row][1]))

    if ticks is not None:
        ax.set_xticks((ticks[col][0], ticks[col][1]))
        if current != "diag":
            ax.set_yticks((ticks[row][0], ticks[row][1]))

    if fig_kwargs["square_subplots"]:
        ax.set_box_aspect(1)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", fig_kwargs["despine"]["offset"]))

    if current == "diag":
        if excl_lower or col == dim - 1 or flat:
            _format_axis(ax, xhide=False, xlabel=labels_dim[col], yhide=True,
                         tickformatter=fig_kwargs["tickformatter"])
        else:
            _format_axis(ax, xhide=True, yhide=True)
    else:
        if row == dim - 1:
            _format_axis(ax, xhide=False, xlabel=labels_dim[col], yhide=True,
                         tickformatter=fig_kwargs["tickformatter"])
        else:
            _format_axis(ax, xhide=True, yhide=True)

    if fig_kwargs["tick_labels"] is not None:
        ax.set_xticklabels((str(fig_kwargs["tick_labels"][col][0]),
                            str(fig_kwargs["tick_labels"][col][1])))

# ---- grid arranger (no torch)

def _arrange_grid_no_torch(
    diag_funcs: List[Optional[Callable]],
    upper_funcs: List[Optional[Callable]],
    lower_funcs: List[Optional[Callable]],
    diag_kwargs: List[Optional[Dict]],
    upper_kwargs: List[Optional[Dict]],
    lower_kwargs: List[Optional[Dict]],
    samples: List[np.ndarray],
    points: List[np.ndarray],
    limits: np.ndarray,
    subset: Optional[List[int]],
    figsize: Optional[Tuple],
    labels: Optional[List[str]],
    ticks: Optional[Union[List, np.ndarray]],
    fig: Optional[FigureBase],
    axes,  # can be None or Axes array
    fig_kwargs: Dict,
) -> Tuple[FigureBase, Axes]:
    dim = samples[0].shape[1]
    if labels is None or labels == []:
        labels = [f"dim {i+1}" for i in range(dim)]
    if ticks is not None and len(ticks) == 1:
        ticks = [ticks[0] for _ in range(dim)]

    if subset is None:
        rows = cols = dim
        subset = list(range(dim))
    else:
        if isinstance(subset, int):
            subset = [subset]
        rows = cols = len(subset)

    excl_lower = all(v is None for v in lower_funcs)
    excl_upper = all(v is None for v in upper_funcs)
    excl_diag  = all(v is None for v in diag_funcs)
    flat = excl_lower and excl_upper
    one_dim = dim == 1

    subset_rows = [1] if flat else subset
    subset_cols = subset

    if fig is None or axes is None:
        fig, axes = plt.subplots(1 if flat else rows, cols, figsize=figsize, **fig_kwargs["subplots"])
    fig.subplots_adjust(**fig_kwargs["fig_subplots_adjust"])
    fig.suptitle(fig_kwargs["title"], **fig_kwargs["title_format"])

    for r_idx, r in enumerate(subset_rows):
        for c_idx, c in enumerate(subset_cols):
            current = "diag" if (flat or r == c) else ("upper" if r < c else "lower")
            if one_dim:
                ax = axes
            elif flat:
                ax = axes[c_idx]
            else:
                ax = axes[r_idx, c_idx]

            if current in fig_kwargs["fig_bg_colors"] and fig_kwargs["fig_bg_colors"][current] is not None:
                ax.set_facecolor(fig_kwargs["fig_bg_colors"][current])

            _format_subplot_np(
                ax, current, limits, ticks, labels, fig_kwargs,
                r, c, dim, flat, excl_lower
            )

            # Diagonal
            if current == "diag":
                if excl_diag:
                    ax.axis("off")
                else:
                    for si, s in enumerate(samples):
                        f = diag_funcs[si]
                        if callable(f):
                            f(ax, s[:, r if flat else c], limits[c], diag_kwargs[si])
                if len(points) > 0:
                    ymin, ymax = ax.get_ylim()
                    for n, p in enumerate(points):
                        ax.plot([p[:, c], p[:, c]], [ymin, ymax],
                                color=fig_kwargs["points_colors"][n],
                                **fig_kwargs["points_diag"],
                                label=fig_kwargs["points_labels"][n])
                if fig_kwargs["legend"] and c == 0:
                    ax.legend(**fig_kwargs["legend_kwargs"])

            # Upper triangle
            elif current == "upper":
                if excl_upper:
                    ax.axis("off")
                else:
                    for si, s in enumerate(samples):
                        f = upper_funcs[si]
                        if callable(f):
                            f(ax, s[:, c], s[:, r], limits[c], limits[r], upper_kwargs[si])
                    if len(points) > 0:
                        for n, p in enumerate(points):
                            ax.plot(p[:, c], p[:, r],
                                    color=fig_kwargs["points_colors"][n],
                                    **fig_kwargs["points_offdiag"])

            # Lower triangle
            elif current == "lower":
                if excl_lower:
                    ax.axis("off")
                else:
                    for si, s in enumerate(samples):
                        f = lower_funcs[si]
                        if callable(f):
                            f(ax, s[:, r], s[:, c], limits[r], limits[c], lower_kwargs[si])
                    if len(points) > 0:
                        for n, p in enumerate(points):
                            ax.plot(p[:, c], p[:, r],
                                    color=fig_kwargs["points_colors"][n],
                                    **fig_kwargs["points_offdiag"])

    # Ellipses if subset smaller than dim
    if len(subset) < dim:
        if flat:
            ax = axes[len(subset) - 1]
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            ax.text(x1 + (x1 - x0)/8.0, (y0 + y1)/2.0, "...",
                    fontsize=plt.rcParams["font.size"] * 2.0)
        else:
            for rr in range(len(subset)):
                ax = axes[rr, len(subset) - 1]
                x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
                ax.text(x1 + (x1 - x0)/8.0, (y0 + y1)/2.0, "...",
                        fontsize=plt.rcParams["font.size"] * 2.0)
                if rr == len(subset) - 1:
                    ax.text(x1 + (x1 - x0)/12.0, y0 - (y1 - y0)/1.5, "...",
                            rotation=-45, fontsize=plt.rcParams["font.size"] * 2.0)

    return fig, axes

# ---- public API: pairplot_numpy (no torch)

def pairplot_numpy(
    samples: Union[List[np.ndarray], np.ndarray, List[List[float]]],
    points: Optional[Union[List[np.ndarray], np.ndarray, List[List[float]]]] = None,
    limits: Optional[Union[List, np.ndarray]] = None,
    subset: Optional[List[int]] = None,
    upper: Optional[Union[List[Optional[str]], str]] = "hist",
    lower: Optional[Union[List[Optional[str]], str]] = None,
    diag: Optional[Union[List[Optional[str]], str]] = "hist",
    figsize: Tuple = (10, 10),
    labels: Optional[List[str]] = None,
    ticks: Optional[Union[List, np.ndarray]] = None,
    offdiag: Optional[Union[List[Optional[str]], str]] = None,
    diag_kwargs: Optional[Union[List[Optional[Dict]], Dict]] = None,
    upper_kwargs: Optional[Union[List[Optional[Dict]], Dict]] = None,
    lower_kwargs: Optional[Union[List[Optional[Dict]], Dict]] = None,
    fig_kwargs: Optional[Dict] = None,
    fig: Optional[FigureBase] = None,
    axes: Optional[Axes] = None,
    **kwargs: Optional[Any],
) -> Tuple[FigureBase, Axes]:
    """
    NumPy-only version of `pairplot` with the same interface/behavior, no torch needed.
    """
    # Back-compat alias
    if offdiag is not None:
        upper = offdiag

    samples, dim, limits_arr, points_list = _prepare_for_plot_no_torch(samples, limits, points)

    # fig kwargs
    fig_kwargs_filled = _get_default_fig_kwargs_no_torch()
    fig_kwargs_filled = _update_dict(fig_kwargs_filled, fig_kwargs)

    # checks
    if fig_kwargs_filled["legend"]:
        assert len(fig_kwargs_filled["samples_labels"]) >= len(samples), \
            "Provide at least as many labels as samples."

    # diag prep
    diag_list = _to_list_string(diag, len(samples))
    diag_kwargs_list = _to_list_kwargs(diag_kwargs, len(samples))
    diag_funcs = _get_diag_funcs_no_torch(diag_list)
    diag_kwargs_filled = []
    for i, (di, dki) in enumerate(zip(diag_list, diag_kwargs_list, strict=False)):
        df = _get_default_diag_kwargs_no_torch(di, i)
        df = _update_dict(df, dki)
        diag_kwargs_filled.append(df)

    # upper prep
    upper_list = _to_list_string(upper, len(samples))
    upper_kwargs_list = _to_list_kwargs(upper_kwargs, len(samples))
    upper_funcs = _get_offdiag_funcs_no_torch(upper_list)
    upper_kwargs_filled = []
    for i, (ui, uki) in enumerate(zip(upper_list, upper_kwargs_list, strict=False)):
        uf = _get_default_offdiag_kwargs_no_torch(ui, i)
        uf = _update_dict(uf, uki)
        upper_kwargs_filled.append(uf)

    # lower prep
    lower_list = _to_list_string(lower, len(samples))
    lower_kwargs_list = _to_list_kwargs(lower_kwargs, len(samples))
    lower_funcs = _get_offdiag_funcs_no_torch(lower_list)
    lower_kwargs_filled = []
    for i, (li, lki) in enumerate(zip(lower_list, lower_kwargs_list, strict=False)):
        lf = _get_default_offdiag_kwargs_no_torch(li, i)
        lf = _update_dict(lf, lki)
        lower_kwargs_filled.append(lf)

    return _arrange_grid_no_torch(
        diag_funcs, upper_funcs, lower_funcs,
        diag_kwargs_filled, upper_kwargs_filled, lower_kwargs_filled,
        samples, points_list, limits_arr, subset, figsize, labels, ticks,
        fig, axes, fig_kwargs_filled
    )
