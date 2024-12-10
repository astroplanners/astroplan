# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Astroplan documentation build configuration file.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this file.
#
# All configuration values have a default. Some values are defined in
# the global Astropy configuration which is loaded here before anything else.
# See astropy_sphinx.conf for which values are set there.

import sys
import datetime

import astroplan

try:
    from sphinx_astropy.conf.v2 import *  # noqa
except ImportError:
    print('ERROR: the documentation requires the sphinx-astropy package to '
          'be installed')
    sys.exit(1)

# -- General configuration ----------------------------------------------------

# Extend astropy intersphinx_mapping with packages we use here
intersphinx_mapping['astroquery'] = ('http://astroquery.readthedocs.io/en/latest/', None)  # noqa: F405 E501

# Exclude astropy intersphinx_mapping for unused packages
del intersphinx_mapping['h5py']  # noqa: F405

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns.append('_templates')  # noqa
# Exclude template PSF block specification documentation
exclude_patterns.append('psf_spec/*')  # noqa

plot_formats = ['png', 'hires.png', 'pdf', 'svg']

# This is added to the end of RST files - a good place to put
# substitutions to be used globally.
rst_epilog = """
.. _Astropy: https://www.astropy.org/
"""

# -- Project information ------------------------------------------------------

# This does not *have* to match the package name, but typically does
project = "astroplan"
author = "Astroplan developers"
copyright = f"{datetime.datetime.now().year}, {author}"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.

# The full version, including alpha/beta/rc tags.
release = astroplan.__version__
# The short X.Y version.
version = ".".join(release.split(".")[:2])

# Only include dev docs in dev version.
dev = "dev" in release

# -- Options for HTML output ---------------------------------------------------

# A NOTE ON HTML THEMES
# The global astropy configuration uses a custom theme
# which is installed along with astropy.

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f'{project} v{release}'

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# Prefixes that are ignored for sorting the Python module index
modindex_common_prefix = ["astroplan."]

# -- Options for LaTeX output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [('index', project + '.tex', project + u' Documentation',
                    author, 'manual')]


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [('index', project.lower(), project + u' Documentation',
              [author], 1)]

# -- Add additional Sphinx extensions -----------------------------------------

# Add additional Sphinx extensions:
extensions += [  # noqa: F405
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.graphviz'
]

# -- Resolving issue number to links in changelog -----------------------------
github_issues_url = 'https://github.com/astropy/astroplan/issues/'

# -- Options for linkcheck output -------------------------------------------
linkcheck_retry = 5
linkcheck_ignore = [
    r'https://github\.com/astropy/astroplan/(?:issues|pull)/\d+',
]
linkcheck_timeout = 180
linkcheck_anchors = False

# -- Turn on nitpicky mode for sphinx (to warn about references not found) ----
nitpicky = True

html_copy_source = False

html_theme_options.update(  # noqa: F405
    {
        "github_url": "https://github.com/astropy/astroplan",
        "external_links": [
            {"name": "astropy docs", "url": "https://docs.astropy.org/en/stable/"},
        ],
        "use_edit_page_button": True,
    }
)

html_context = {
    "default_mode": "light",
    "to_be_indexed": ["stable", "latest"],
    "is_development": dev,
    "github_user": "astropy",
    "github_repo": "astroplan",
    "github_version": "main",
    "doc_path": "docs",
}

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
html_extra_path = ["robots.txt"]
