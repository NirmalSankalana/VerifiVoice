import sphinx_rtd_theme


html_theme = 'alabaster'
html_static_path = ['_static']


extensions = [
  "sphinx_rtd_theme","sphinx.ext.autodoc"]

html_theme = "sphinx_rtd_theme"



# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VerifyVoice'
copyright = '2024, Nirmal Sankalana, Nipun Thejan'
author = 'Nirmal Sankalana, Nipun Thejan'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

