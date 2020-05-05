from setuptools_scm import get_version
try:
    
    __version__ = get_version()
except Exception:
    # package is not installed
    __version__ = "0.0.0"
    pass