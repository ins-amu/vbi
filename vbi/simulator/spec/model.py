from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np


@dataclass(frozen=True)
class StateVar:
    name: str
    default_init: float = 0.0
    noise: bool = False
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass(frozen=True)
class Parameter:
    name: str
    default: float | np.ndarray
    description: str = ""


def _fmt_default(v) -> str:
    """Format a parameter default value for display."""
    if isinstance(v, np.ndarray):
        return f"array({v.shape})" if v.ndim > 0 else f"{float(v):.4g}"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


@dataclass(frozen=True)
class ModelSpec:
    """
    Backend-agnostic description of a neural mass model.

    dfun_str maps each state variable name to a bare math expression string.
    Only these symbols may appear: state variable names, parameter names, 'c'
    (coupling input, scalar per node), and the math functions:
        exp, log, sin, cos, tanh, sqrt, abs, pi
    No 'np.' prefix - the code generator injects the namespace.

    dfun_latex maps each state variable name to a LaTeX string for the RHS
    of its differential equation.  Used by describe() to render equations in
    Jupyter notebooks.  Falls back to dfun_str if not provided.
    """
    name: str
    state_variables: tuple[StateVar, ...]
    parameters: tuple[Parameter, ...]
    cvar: tuple[str, ...]         # names of coupling-variable state vars
    dfun_str: dict[str, str]      # {sv_name: expression_string}
    noise_variables: tuple[str, ...] = ()
    reference: str = ""
    dfun_latex: dict[str, str] = field(default_factory=dict)
    # Optional LaTeX block rendered below the equations (definitions, where-clauses).
    latex_notes: str = ""

    @property
    def sv_names(self) -> tuple[str, ...]:
        return tuple(sv.name for sv in self.state_variables)

    @property
    def n_sv(self) -> int:
        return len(self.state_variables)

    @property
    def cvar_indices(self) -> tuple[int, ...]:
        names = self.sv_names
        return tuple(names.index(c) for c in self.cvar)

    @property
    def noise_indices(self) -> tuple[int, ...]:
        names = self.sv_names
        return tuple(names.index(n) for n in self.noise_variables)

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    @property
    def default_params(self) -> dict[str, float | np.ndarray]:
        return {p.name: p.default for p in self.parameters}

    def describe(self) -> None:
        """
        Render model equations, parameters, and state variables.

        In Jupyter notebooks the output is typeset with MathJax (Markdown +
        LaTeX).  In a plain Python session it prints a text summary instead.

        LaTeX equations use the strings stored in ``dfun_latex``; models that
        do not provide ``dfun_latex`` fall back to the raw ``dfun_str``
        expressions wrapped in a code block.

        Example
        -------
        >>> from vbi.simulator.models import mpr
        >>> mpr.describe()
        """
        def _in_notebook() -> bool:
            try:
                from IPython import get_ipython
                shell = get_ipython()
                return shell is not None and shell.__class__.__name__ in (
                    "ZMQInteractiveShell", "TerminalInteractiveShell"
                )
            except ImportError:
                return False

        in_nb = _in_notebook()

        if in_nb:
            from IPython.display import display, Markdown
            def _render(md: str) -> None:
                display(Markdown(md))
        else:
            def _render(md: str) -> None:
                # Plain-text fallback: strip markdown/LaTeX decoration
                import re
                txt = md
                txt = re.sub(r"\$\$(.+?)\$\$", lambda m: f"  {m.group(1)}", txt, flags=re.S)
                txt = txt.replace("**", "").replace("`", "")
                print(txt)

        lines: list[str] = []

        # --- Title & reference ---
        lines += [f"## {self.name}", ""]
        if self.reference:
            lines += [f"*{self.reference}*", ""]

        # --- Equations ---
        lines += ["---", "**Equations**", ""]
        has_latex = bool(self.dfun_latex)
        if has_latex and in_nb:
            # Group all equations into a left-aligned aligned block
            eq_rows = []
            for sv_name in self.sv_names:
                rhs = self.dfun_latex.get(sv_name, self.dfun_str[sv_name])
                eq_rows.append(f"\\dot{{{sv_name}}} &= {rhs}")
            eq_body = " \\\\\n".join(eq_rows)
            eq_block = (
                '<div style="text-align: left; padding: 0.4em 0">\n\n'
                f"$$\n\\begin{{aligned}}\n{eq_body}\n\\end{{aligned}}\n$$\n\n"
                "</div>"
            )
            lines += [eq_block, ""]
        else:
            for sv_name in self.sv_names:
                if has_latex and sv_name in self.dfun_latex:
                    rhs = self.dfun_latex[sv_name]
                    lines.append(f"$$\\dot{{{sv_name}}} = {rhs}$$")
                else:
                    rhs = self.dfun_str[sv_name]
                    lines.append(f"**d{sv_name}/dt** = `{rhs}`")
                lines.append("")

        # --- Where-clause / supplementary notes ---
        if self.latex_notes:
            lines += ["**where**", "", self.latex_notes, ""]

        # --- Coupling ---
        coup = ", ".join(f"`{c}`" for c in self.cvar)
        lines += [f"**Coupling variables:** {coup}", ""]

        # --- State variables table ---
        lines += ["---", "**State variables**", ""]
        lines += ["| Name | Init | Noise | Bounds |",
                  "|------|:----:|:-----:|--------|"]
        for sv in self.state_variables:
            noise = "✓" if sv.noise else "-"
            lo = str(sv.lower_bound) if sv.lower_bound is not None else "−∞"
            hi = str(sv.upper_bound) if sv.upper_bound is not None else "+∞"
            lines.append(f"| `{sv.name}` | {sv.default_init} | {noise} | [{lo}, {hi}] |")
        lines.append("")

        # --- Parameters table ---
        lines += ["---", "**Parameters**", ""]
        lines += ["| Name | Default | Description |",
                  "|------|:-------:|-------------|"]
        for p in self.parameters:
            lines.append(f"| `{p.name}` | {_fmt_default(p.default)} | {p.description} |")

        _render("\n".join(lines))
