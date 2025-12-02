import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'MiniNumPy'
copyright = '2025, Duc Hung NGUYEN'
author = 'Duc Hung NGUYEN'
release = '0.0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_autodoc_typehints",
]

# Generate autosummary .rst files automatically
autosummary_generate = True

# Templates
templates_path = ['_templates']
exclude_patterns = []

# Autodoc settings
autodoc_member_order = 'bysource'

autodoc_typehints = "description"
typehints_fully_qualified = True
always_document_param_types = True

# Show source (viewcode) but hide the ugly Sphinx default source link
html_show_sourcelink = False  

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__, __str__, __add__, __sub__, __mul__, __truediv__, __pow__, __matmul__",
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "text": "MiniNumPy"
    },
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}

html_static_path = ['_static']
