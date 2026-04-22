"""Sphinx configuration for ``ooai_llm`` documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime, UTC
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

project = "ooai-llm"
author = "OpenAI"
copyright = f"{datetime.now(UTC).year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_inline_tabs",
    "sphinxcontrib.mermaid",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.autodoc_pydantic",
    "autoapi.extension",
    "sphinxext.opengraph",
    "notfound.extension",
    "sphinx_last_updated_by_git",
    "sphinx_reredirects",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "https://ooai-llm.readthedocs.io/")
html_theme_options = {
    "source_repository": "https://github.com/pr1m8/ooai-llm/",
    "source_branch": "main",
    "source_directory": "docs/",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]

try:
    release = package_version("ooai-llm")
except PackageNotFoundError:
    release = "0.0.0"
version = release

autoapi_type = "python"
autoapi_dirs = [str(SRC / "ooai_llm")]
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]

autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = True
autodoc_pydantic_field_show_default = True

typehints_defaults = "comma"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

ogp_site_url = html_baseurl
ogp_site_name = project
ogp_use_first_image = True

notfound_urls_prefix = "/en/latest/"

redirects = {
    "api": "api/index.html",
}

spelling_lang = "en_US"
spelling_show_suggestions = True
spelling_warning = True
