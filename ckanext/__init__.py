# This is a namespace package (PEP 420)
# Modern Python 3 uses implicit namespace packages
# This file should remain empty or minimal
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
