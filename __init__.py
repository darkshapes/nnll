from importlib.metadata import version, PackageNotFoundError
import _version as v

print(v.version)
print(v.version_tuple)
try:
    __version__ = version("nnll")
except PackageNotFoundError:
    # package is not installed
    pass
