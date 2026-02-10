# encoding: utf-8
"""
ckanext-chat

A CKAN extension that adds a pydantic AI chat interface with user-aware context.
"""

# Expose package version from setuptools-scm
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8 fallback
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("ckanext-chat")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"

__all__ = ["__version__"]
