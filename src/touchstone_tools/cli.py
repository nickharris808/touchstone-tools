"""Command line for touchstone-tools.

    touchstone-tools info      measured.s2p
    touchstone-tools convert   measured.s2p out.s2p --to z --format ri
    touchstone-tools renorm    measured.s2p out75.s2p --z0 75

Exit codes:
  0  done
  1  refused: the request has no answer (a singular conversion)
  2  usage, or a file that could not be read or written
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .io import FORMATS, FREQ_UNITS, read_touchstone, write_touchstone
from .network import OHM, TouchstoneError


def _info(net) -> dict:
    s = net.s
    return {
        "file": net.path,
        "n_ports": net.n_ports,
        "n_freq": net.n_freq,
        "kind": net.kind,
        "z0_ohm": net.z0,
        "freq_start_hz": float(net.freq_hz[0]),
        "freq_stop_hz": float(net.freq_hz[-1]),
        "max_abs": float(np.max(np.abs(s))),
        # Reported, not judged. Whether S = S^T *should* hold is a question
        # about the device -- a ferrite isolator is non-reciprocal by design --
        # so this states the measured asymmetry and stops there. sparam-lint is
        # the tool that renders a verdict.
        "max_abs_asymmetry": float(np.max(np.abs(s - s.transpose(0, 2, 1)))),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="touchstone-tools",
        description="Read, write and convert N-port network files.")
    p.add_argument("--version", action="version",
                   version=f"touchstone-tools {__version__}")
    subs = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    i = subs.add_parser("info", help="Describe a file.")
    i.add_argument("path")
    i.add_argument("--json", action="store_true")

    c = subs.add_parser("convert", help="Convert between S, Y and Z, or between formats.")
    c.add_argument("src")
    c.add_argument("dest")
    c.add_argument("--to", choices=["s", "y", "z"], default=None,
                   help="parameter kind to convert to (default: keep)")
    c.add_argument("--format", choices=list(FORMATS), default="ri",
                   dest="fmt", help="number format to write (default ri)")
    c.add_argument("--freq-unit", choices=list(FREQ_UNITS), default="hz")

    r = subs.add_parser("renorm", help="Re-reference to a different impedance.")
    r.add_argument("src")
    r.add_argument("dest")
    r.add_argument("--z0", type=float, required=True, help="new reference impedance, ohms")
    r.add_argument("--format", choices=list(FORMATS), default="ri", dest="fmt")
    r.add_argument("--freq-unit", choices=list(FREQ_UNITS), default="hz")

    args = p.parse_args(argv)

    try:
        net = read_touchstone(args.src if args.cmd != "info" else args.path)
    except TouchstoneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "info":
        d = _info(net)
        if args.json:
            print(json.dumps(d, indent=2))
        else:
            print(f"{Path(d['file']).name}: {d['n_ports']}-port {d['kind'].upper()}, "
                  f"{d['n_freq']} points, "
                  f"{d['freq_start_hz']/1e9:.4g}-{d['freq_stop_hz']/1e9:.4g} GHz, "
                  f"z0={d['z0_ohm']:g}{OHM}")
            print(f"  max |entry|        {d['max_abs']:.6g}")
            print(f"  max |S - S^T|      {d['max_abs_asymmetry']:.3e}  "
                  "(reported, not judged)")
        return 0

    try:
        if args.cmd == "renorm":
            out = net.renormalized(args.z0)
        else:
            # `--to` is only defined on convert; renorm never reaches here.
            out = net.to(args.to) if getattr(args, "to", None) else net
    except TouchstoneError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    try:
        written = write_touchstone(out, args.dest, fmt=args.fmt,
                                   freq_unit=args.freq_unit)
    except TouchstoneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {written}  ({out.n_ports}-port {out.kind.upper()}, "
          f"{out.n_freq} points, z0={out.z0:g}{OHM}, {args.fmt.upper()})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
