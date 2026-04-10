from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nn-pytorch")
except PackageNotFoundError:
    __version__ = "unknown"