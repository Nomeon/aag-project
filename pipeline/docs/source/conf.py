# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'aag-project'
copyright = '2024, Bayir, Borger, Friedrichs, Mählmann, McCarthy, Nijhuis, Nikolarakis'
author = 'Bayir, Borger, Friedrichs, Mählmann, McCarthy, Nijhuis, Nikolarakis'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx_rtd_theme', 'sphinx.ext.coverage']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
html_logo = 'alliance.png'
html_theme_options = {
  'logo_only': True,
  'display_version': False, 
}
html_show_sphinx = False