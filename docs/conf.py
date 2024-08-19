import sys
import os
import sphinx_bootstrap_theme
from setuptools_scm import get_version

sys.path.insert(0, os.path.abspath("../vbi"))

needs_sphinx = "1.3"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.viewcode",
    # 'nbsphinx'
]

source_suffix = ".rst"
master_doc = "index"
project = "vbi"
copyright = "2023, Abolfazl Ziaeemehr"
release = version = get_version(root="..", relative_to=__file__)

default_role = "any"
add_function_parentheses = True
add_module_names = False
html_theme = 'nature'
# hrml_theme = "sphinx_pdj_theme"

# html_theme = "bootstrap"
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

pygments_style = "colorful"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

numpydoc_show_class_members = False
autodoc_member_order = "bysource"
graphviz_output_format = "svg"
toc_object_entries_show_parents = "hide"


def on_missing_reference(app, env, node, contnode):
    if node["reftype"] == "any":
        return contnode
    else:
        return None


def setup(app):
    app.connect("missing-reference", on_missing_reference)
