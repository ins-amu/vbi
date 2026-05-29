# Documentation Milestone — sphinx-gallery + RTD Versioning

## Goal

Provide MNE-style documentation where every demo/tutorial:
- Renders executed outputs (plots, tables) inline in the HTML docs
- Offers a "Download `.py`" and "Download `.ipynb`" button per page
- Has a "Open in Colab" badge that works without committing notebooks to git
- Lives in clean `.py` scripts that produce readable git diffs

Reference: https://mne.tools/stable/auto_tutorials/inverse/10_stc_class.html

---

## Chosen Stack

| Tool | Role |
|---|---|
| `sphinx-gallery` | Executes `.py` scripts during docs build, renders outputs inline, auto-generates `.ipynb` and download buttons |
| `ju2py` (`pip install ju2py`) | Converts existing `.ipynb` notebooks to clean sphinx-gallery-compatible `.py` scripts |
| `pydata-sphinx-theme` | HTML theme — same as MNE, NumPy, SciPy, pandas. Replaces `furo`. Has built-in version switcher dropdown (latest/stable), sphinx-gallery thumbnail grid, and top navigation bar. |
| ReadTheDocs | Hosts `latest` (main branch) and `stable` (latest git tag) versions |

Jupytext was considered and rejected: sphinx-gallery solves discoverability,
Colab, docs rendering, and git cleanliness all at once, matching the target
reference exactly.

---

## RTD Versioning Strategy

| RTD version | Source | Notebooks executed? |
|---|---|---|
| `latest` | `main` branch | Yes (during RTD build) |
| `stable` | latest git tag | Yes (during RTD build) |
| feature branches | any other branch | Not built / not published |

No `dev` docs version. `latest` already serves that role.

### How RTD knows which versions to build

Configured in the RTD project dashboard (not in `.readthedocs.yaml`):
- Enable "Automation Rules" → build tags matching `v*` as new versions
- RTD auto-designates the newest tag as `stable`
- Default branch (`main`) is always `latest`

---

## Steps

### Step 1 — Add root-level symlink for discoverability

Users expect `examples/` at repo root. Sphinx resolves files at `docs/examples/`.
A symlink satisfies both without moving anything.

```bash
ln -s docs/examples examples
git add examples
git commit -m "docs: add examples/ symlink for discoverability"
```

### Step 2 — Switch to pydata-sphinx-theme

Add to `docs/requirements.txt`:
```
pydata-sphinx-theme
```

Remove `furo` from `docs/requirements.txt`.

In `docs/conf.py` replace:
```python
html_theme = "furo"
html_theme_options = {
    "default_mode": "light",
}
```
with:
```python
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["version-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": "https://<your-rtd-project>.readthedocs.io/en/latest/_static/switcher.json",
        "version_match": version,
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/<org>/vbi",
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {
        "text": "vbi",
    },
}
```

The `switcher.json` file (placed in `docs/_static/`) lists available doc versions
so users can switch between `latest` and `stable` from a dropdown — identical to
what MNE shows. It is a small static JSON file that must be updated at each release:

```json
[
  {"name": "stable (v0.x.x)", "version": "stable", "url": "https://<project>.readthedocs.io/en/stable/"},
  {"name": "latest",          "version": "latest",  "url": "https://<project>.readthedocs.io/en/latest/"}
]
```

### Step 3 — Install and configure sphinx-gallery

Add to `docs/requirements.txt`:
```
sphinx-gallery
Pillow          # required by sphinx-gallery for thumbnail generation
```

Update `docs/conf.py`:
```python
extensions = [
    ...
    "sphinx_gallery.gen_gallery",   # replaces nbsphinx for demo pages
]

sphinx_gallery_conf = {
    "examples_dirs":   ["../docs/examples/simulator_models",
                        "../docs/examples/inference"],  # source .py dirs
    "gallery_dirs":    ["auto_examples/simulator_models",
                        "auto_examples/inference"],     # output HTML dirs
    "filename_pattern": r".*_demo\.py|.*_tutorial\.py|.*benchmark.*\.py",
    "ignore_pattern":   r"helpers\.py|__init__\.py",
    "execute_files":    True,                            # True for main/stable
    "plot_gallery":     True,
    "download_all_examples": True,                       # enables .ipynb button
    "first_notebook_cell": (
        "# This notebook is auto-generated from vbi documentation.\n"
        "# Install: pip install vbi\n"
    ),
    # Colab badge — injected into every generated notebook
    "binder_links": False,   # use Colab instead (see first_notebook_cell)
    "thumbnail_size": (400, 280),
    # For scripts that cannot run in RTD (CUDA, heavy C++):
    # add  # sphinx_gallery_dummy_images = 1  at top of those scripts
    # and place a pre-saved image in outputs/ with the same stem
}
```

Remove or guard `nbsphinx_execute = 'never'` — it is no longer needed once
`nbsphinx` is replaced by `sphinx_gallery`.

### Step 4 — Format existing `.py` scripts for sphinx-gallery

sphinx-gallery requires a module-level RST docstring at the top of each script.
Minimal change to existing scripts:

```python
"""
Kuramoto Model Demo
===================

Demonstrates the Kuramoto oscillator across numpy, numba, and C++ backends.
Benchmarks wallclock time and validates output agreement.
"""

# %%
# Setup
# -----
import numpy as np
from vbi import ...
```

`# %%` marks section boundaries (rendered as headed subsections in HTML).
Everything else in the script is unchanged.

### Step 5 — Convert old notebooks using ju2py

Old notebooks live in `docs/examples/*.ipynb` (legacy API). Convert them to
sphinx-gallery-compatible scripts before editing manually:

```bash
pip install ju2py
ju2py docs/examples/mpr_sde_numba.ipynb          # produces mpr_sde_numba.py
ju2py docs/examples/jansen_rit_sde_numba.ipynb
# ... repeat for each notebook to migrate
```

Then:
1. If the notebook is being updated to the new API → place the generated script
   in `docs/examples/simulator_models/` and add the module docstring + `# %%` markers
2. If the notebook is not yet being migrated → leave it untouched in `docs/examples/`
3. Only after the new-API script is verified and live in the docs, add a deprecation
   notice to the old notebook and move it to `docs/examples/legacy/`

**Important:** do not batch-migrate notebooks. Each notebook moves individually
when its new-API equivalent is complete and tested. The `legacy/` folder is
created only when the first notebook is actually moved there.

### Step 6 — Handle CUDA and heavy C++ demos

These cannot execute in RTD's build environment (no GPU). For each such script:

```python
"""
CUDA Sweep Demo
===============

...
"""
# sphinx_gallery_dummy_images = 1
```

Place a pre-saved representative output image at
`docs/examples/simulator_models/outputs/cuda_sweep_demo.png`.
sphinx-gallery will use it as the gallery thumbnail without executing the script.

### Step 7 — Update .readthedocs.yaml

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.10"
  apt_packages:
    - swig
    - build-essential

sphinx:
  configuration: docs/conf.py

python:
  install:
  - method: pip
    path: .
  - requirements: docs/requirements.txt
```

No conditional execution logic needed in the YAML — sphinx-gallery handles
execution at build time. CUDA-only scripts are opted out via the script-level
comment in Step 5.

### Step 8 — Update index.rst to include gallery pages

```rst
.. toctree::
   :maxdepth: 2

   auto_examples/simulator_models/index
   auto_examples/inference/index
```

Keep existing `Examples.rst` / `nbsphinx` references to old notebooks intact.
Both systems coexist: sphinx-gallery handles new scripts, nbsphinx handles old
notebooks. Remove an old notebook's entry from `Examples.rst` only after it has
been migrated and its new gallery page is live.

### Step 9 — Coexistence structure (gradual, not a one-time reorganization)

At any point in time the layout will look like:

```
docs/examples/
  simulator_models/   ← new pipeline demos (sphinx-gallery .py scripts)
  inference/          ← inference demos (sphinx-gallery .py scripts)
  legacy/             ← created only when first notebook is retired
  *.ipynb             ← old notebooks, stay here until individually migrated
```

When a specific old notebook is ready to retire:
1. Add to the top cell: `# ⚠️ This notebook uses the old vbi API. See <link> for the current version.`
2. Move the `.ipynb` to `docs/examples/legacy/`
3. Remove its entry from `Examples.rst`, add a redirect note pointing to the new gallery page
4. Repeat per notebook at your own pace — no deadline, no batch operation

### Step 10 — Validate locally before pushing to main

```bash
cd docs
make clean
make html
# Open _build/html/auto_examples/simulator_models/index.html
```

Check:
- Outputs render inline
- Download buttons appear (`.py` and `.ipynb`)
- Thumbnails show in gallery index
- CUDA scripts show pre-saved image as thumbnail

---

## Migration Priority Order

Migration is intentionally gradual — no hard deadline to retire old notebooks.

1. `simulator_models/` demo scripts — already `.py`, add docstring + `# %%` markers (lowest friction, do first)
2. `benchmark_*.py` scripts — same as above
3. High-value, frequently referenced old notebooks (mpr, jansen_rit, kuramoto) — convert with `ju2py` when the new API for that model is stable
4. CUDA/CuPy notebooks — mark dummy, keep pre-saved output images
5. Remaining old notebooks — retire individually as new-API equivalents are written; no need to rush

---

## Open Questions / Decisions Deferred

- Whether to add a Colab badge per-page via `first_notebook_cell` or via a
  sphinx extension (sphinx-gallery supports a `binder_url_fct` hook that can
  be adapted for Colab).
- Whether `docs/examples/legacy/` notebooks should be removed entirely after
  one release cycle or kept indefinitely — decided per notebook, no blanket policy.
- Whether inference demos (APT, MCMC, SBI) go in the same gallery or a
  separate one — depends on how heavy they are to execute on RTD.
