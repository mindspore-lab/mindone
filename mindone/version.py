from importlib.metadata import version

try:
    __version__ = version("mindone")
except ModuleNotFoundError:
    __version__ = "0.3.0.dev0"
