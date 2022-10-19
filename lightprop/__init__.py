import os

from .io.io import load

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"  # disabling tensorflow Info and warning logger

__all__ = ("load",)
