# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "NeuroStatsLab"
copyright = "2024, SJ Venditto"
author = "SJ Venditto"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["mycss.css"]
# html_js_files = ["myjs.js"]

html_theme_options = {
    "logo": {
        "text": [],
        "image_light": "_static/NSL_logo.png",
        "image_dark": "_static/NSL_logo.png",
    },
    "secondary_sidebar_items": {
        "**": ["page-toc"],
        "index": [],
    },
}
