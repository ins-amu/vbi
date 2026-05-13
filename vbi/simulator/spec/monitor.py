from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MonitorSpec:
    """
    Specifies one recording channel for the simulator.

    Follows TVB monitor conventions:
      raw       — every integration step (large; use for debugging)
      subsample — every `period` ms (decimation, TVB SubSample)
      tavg      — time-average over `period` ms windows (TVB TemporalAverage);
                  preferred for SBI — acts as a low-pass filter
      gavg      — spatial mean of VOIs, every `period` ms (TVB GlobalAverage)
      bold      — Balloon-Windkessel BOLD, output at `tr` ms (TVB Bold)

    Parameters
    ----------
    kind : str
    period : float | None
        Sampling period in ms. Ignored for 'raw' (uses dt). Required for
        'subsample', 'tavg', 'gavg'. Ignored for 'bold' (uses tr).
    variables : tuple[str, ...]
        State-variable names to record. Empty = all model VOIs.
    tr : float
        BOLD repetition time in ms (only for kind='bold').
    """
    kind: Literal["raw", "subsample", "tavg", "gavg", "bold"]
    period: float | None = None
    variables: tuple[str, ...] = ()
    tr: float = 2000.0              # ms — only used when kind == "bold"
