"""SSTV Decoder: decodes slow-scan TV pictures from any audio playing on this PC."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sstv-decoder")
except PackageNotFoundError:  # running from a checkout without installing
    __version__ = "0.0.0+checkout"
APP_ID = "io.github.ssjrocks.SSTVDecoder"
APP_NAME = "SSTV Decoder"
