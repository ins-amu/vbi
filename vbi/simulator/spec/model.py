from __future__ import annotations
import dataclasses
import html
import re
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

    def with_init(self, **inits: float | np.ndarray) -> "ModelSpec":
        """
        Return a copy with the named state variables' initial conditions
        overridden.

        Example
        -------
        >>> kuramoto_with_init = kuramoto.with_init(theta=theta0)
        """
        unknown = set(inits) - set(self.sv_names)
        if unknown:
            raise ValueError(
                f"Unknown state variable(s): {sorted(unknown)}. "
                f"Available: {self.sv_names}"
            )
        new_state_variables = tuple(
            dataclasses.replace(sv, default_init=inits[sv.name])
            if sv.name in inits else sv
            for sv in self.state_variables
        )
        return dataclasses.replace(self, state_variables=new_state_variables)

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

    def _repr_html_(self) -> str:
        """
        Rich HTML representation, used automatically when a bare ``ModelSpec``
        is the last expression in a Jupyter cell or a sphinx-gallery code
        block.  Equations use TeX display-math delimiters (``\\[ ... \\]``)
        so MathJax renders them client-side in both venues.
        """
        def _inline(text: str) -> str:
            # Minimal inline markdown -> HTML: **bold** and `code` spans.
            # Math delimiters ($...$) are left untouched for MathJax.
            text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
            text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
            return text

        parts: list[str] = [f"<h3>{html.escape(self.name)}</h3>"]
        if self.reference:
            parts.append(f"<p><em>{html.escape(self.reference)}</em></p>")

        # --- Equations ---
        parts.append("<p><strong>Equations</strong></p>")
        eq_rows = [
            f"\\dot{{{sv_name}}} &= {self.dfun_latex.get(sv_name, self.dfun_str[sv_name])}"
            for sv_name in self.sv_names
        ]
        eq_body = " \\\\\n".join(eq_rows)
        parts.append(f"\\[\n\\begin{{aligned}}\n{eq_body}\n\\end{{aligned}}\n\\]")

        if self.latex_notes:
            parts.append("<p><strong>where</strong></p>")
            parts.append(f"<p>{_inline(self.latex_notes)}</p>")

        coup = ", ".join(f"<code>{html.escape(c)}</code>" for c in self.cvar)
        parts.append(f"<p><strong>Coupling variables:</strong> {coup}</p>")

        # --- State variables table ---
        sv_rows = "".join(
            f"<tr><td><code>{html.escape(sv.name)}</code></td>"
            f"<td>{sv.default_init}</td>"
            f"<td>{'✓' if sv.noise else '-'}</td>"
            f"<td>[{sv.lower_bound if sv.lower_bound is not None else '−∞'}, "
            f"{sv.upper_bound if sv.upper_bound is not None else '+∞'}]</td></tr>"
            for sv in self.state_variables
        )
        parts.append(
            "<p><strong>State variables</strong></p>"
            "<table><thead><tr><th>Name</th><th>Init</th><th>Noise</th>"
            f"<th>Bounds</th></tr></thead><tbody>{sv_rows}</tbody></table>"
        )

        # --- Parameters table ---
        p_rows = "".join(
            f"<tr><td><code>{html.escape(p.name)}</code></td>"
            f"<td>{html.escape(_fmt_default(p.default))}</td>"
            f"<td>{html.escape(p.description)}</td></tr>"
            for p in self.parameters
        )
        parts.append(
            "<p><strong>Parameters</strong></p>"
            "<table><thead><tr><th>Name</th><th>Default</th>"
            f"<th>Description</th></tr></thead><tbody>{p_rows}</tbody></table>"
        )

        return "\n".join(parts)
