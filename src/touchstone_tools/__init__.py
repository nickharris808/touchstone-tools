"""touchstone-tools — read, write and convert N-port network files.

`sparam-lint` tells you a model is broken. This builds one that cannot be:
every write path refuses what the read path would refuse, so a file with NaN
in it, a sweep that runs backwards, or an extension that disagrees with the
port count never leaves this library.

    from touchstone_tools import read_touchstone, write_touchstone

    net = read_touchstone("measured.s2p")
    write_touchstone(net.renormalized(75.0), "measured_75.s2p", fmt="ri")
"""

from __future__ import annotations

__version__ = "0.1.0"

from .io import FORMATS, FREQ_UNITS, read_touchstone, write_touchstone
from .network import (
    Network,
    TouchstoneError,
    renormalize,
    s_to_y,
    s_to_z,
    y_to_s,
    y_to_z,
    z_to_s,
    z_to_y,
)

__all__ = [
    "Network", "TouchstoneError", "read_touchstone", "write_touchstone",
    "FORMATS", "FREQ_UNITS", "renormalize",
    "s_to_z", "z_to_s", "s_to_y", "y_to_s", "z_to_y", "y_to_z",
    "__version__",
]
