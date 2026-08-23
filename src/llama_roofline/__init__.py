"""llama-roofline: is your llama.cpp setup memory-bandwidth-bound? Measure it.

Decode (token generation) in a dense transformer streams the whole model out of memory
once per token, so decode throughput is capped by memory bandwidth, not by compute:

    tok/s  ~=  BW_eff / model_bytes

This package measures both sides of that equation on *your* machine -- the achievable
memory-read ceiling, and the decode/prefill throughput of your own GGUF models via
llama.cpp's ``llama-bench`` -- then reports how close you are to the wall.
"""

__version__ = "0.2.0"
__author__ = "Manu Nicholas Jacob"
__license__ = "MIT"

# Version of the results JSON schema written by `llama-roofline run`.
SCHEMA_VERSION = 1

__all__ = ["__version__", "SCHEMA_VERSION"]
