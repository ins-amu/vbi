import sys
import os
import subprocess
import sphinx_bootstrap_theme
# Import version directly from _version.py instead of using setuptools_scm
sys.path.insert(0, os.path.abspath(".."))
from vbi._version import __version__

needs_sphinx = "0.2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.viewcode",
    'nbsphinx'
]

source_suffix = ".rst"
master_doc = "index"
project = "vbi"
copyright = "2023, Abolfazl Ziaeemehr"
release = version = __version__
nbsphinx_execute = 'never'

default_role = "any"
add_module_names = False
html_theme = 'nature'
pygments_style = "colorful"
add_function_parentheses = True
html_static_path = ['_static']

# html_theme = "bootstrap"
# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

numpydoc_show_class_members = False
autodoc_member_order = "bysource"
graphviz_output_format = "svg"
toc_object_entries_show_parents = "hide"

# Autodoc configuration for better handling of Numba classes
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Custom autodoc processing for Numba jitclass
def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip certain members in autodoc to clean up Numba class documentation."""
    # Skip the class_type attribute that Numba adds
    if name == 'class_type':
        return True
    # Skip internal Numba attributes
    if name.startswith('_numba_'):
        return True
    # Skip other unwanted Numba internal attributes
    if name in ['__dict__', '__weakref__', '_type', '_literal_value']:
        return True
    return skip

def process_docstring(app, what, name, obj, options, lines):
    """Process docstrings to clean up Numba class documentation."""
    if what == "class" and hasattr(obj, 'class_type'):
        # This is a Numba jitclass, clean up the docstring
        if lines and 'jitclass' in str(lines[0]):
            # Replace jitclass type info with cleaner description
            lines[0] = "Numba-compiled parameter class for efficient computation."


def on_missing_reference(app, env, node, contnode):
    if node["reftype"] == "any":
        return contnode
    else:
        return None


def setup(app):
    app.connect("missing-reference", on_missing_reference)
    app.connect("autodoc-skip-member", autodoc_skip_member)
    app.connect("autodoc-process-docstring", process_docstring)

    # cpp_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vbi", "models", "cpp", "_src"))
    # try:
    #     subprocess.run(["make"], cwd=cpp_src, check=True)
    #     app.info("C++ code compiled successfully.")
    # except subprocess.CalledProcessError as err:
    #     app.warn(f"Failed to compile C++ code in {cpp_src}: {err}")
