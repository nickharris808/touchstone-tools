"""Touchstone 1.x read and write.

The one real trap in this format: **two-port files store data in the order
S11 S21 S12 S22** -- column-major -- while three-port and above are row-major
(S11 S12 S13 S21 ...). A reader that assumes row-major everywhere silently
transposes every 2-port file, and a writer that does the same silently
transposes every 2-port file it emits. Both are handled here and both are
pinned by tests with four distinguishable values.

The writer refuses to emit a file it would refuse to read. Every existing
writer in this space will happily put ``NaN`` in a file; this one raises,
because a file containing NaN is a physics verdict waiting to be computed on
nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .network import Network, TouchstoneError

__all__ = ["read_touchstone", "write_touchstone", "FREQ_UNITS", "FORMATS"]

FREQ_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12}
FORMATS = ("ri", "ma", "db")
_KINDS = ("s", "y", "z", "h", "g")


def _parse_option_line(line: str) -> tuple[float, str, str, float]:
    toks = line[1:].split()
    freq_mult, kind, fmt, z0 = 1e9, "s", "ma", 50.0
    i = 0
    while i < len(toks):
        t = toks[i].lower()
        if t in FREQ_UNITS:
            freq_mult = FREQ_UNITS[t]
        elif t in _KINDS:
            kind = t
        elif t in FORMATS:
            fmt = t
        elif t == "r" and i + 1 < len(toks):
            try:
                z0 = float(toks[i + 1])
            except ValueError as exc:
                raise TouchstoneError(
                    f"bad reference impedance {toks[i+1]!r} on the option line"
                ) from exc
            i += 1
        i += 1
    return freq_mult, kind, fmt, z0


def _to_complex_array(a: np.ndarray, b: np.ndarray, fmt: str) -> np.ndarray:
    if fmt == "ri":
        return a + 1j * b
    mag = a if fmt == "ma" else 10.0 ** (a / 20.0)
    return mag * np.exp(1j * np.radians(b))


def _from_complex_array(z: np.ndarray, fmt: str) -> tuple[np.ndarray, np.ndarray]:
    if fmt == "ri":
        return z.real, z.imag
    mag, ang = np.abs(z), np.degrees(np.angle(z))
    if fmt == "ma":
        return mag, ang
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(mag)
    # A true zero is -inf dB. That is the correct value and an unwritable one,
    # so say why rather than emitting "-inf" for another tool to choke on.
    if not np.all(np.isfinite(db)):
        raise TouchstoneError(
            "cannot write DB format: an entry is exactly zero, which is -inf dB. "
            "Use format='ri' or 'ma', which represent zero exactly."
        )
    return db, ang


def read_touchstone(path: str | Path) -> Network:
    """Read a Touchstone 1.x file. Refuses anything it cannot read exactly."""
    path = Path(path)
    if not path.exists():
        raise TouchstoneError(f"no such file: {path}")

    m = re.search(r"\.s(\d+)p$", path.name, re.IGNORECASE)
    if not m:
        raise TouchstoneError(
            f"cannot infer port count from filename {path.name!r} -- Touchstone "
            "encodes it in the extension (.s2p, .s4p, ...)."
        )
    n_ports = int(m.group(1))
    if n_ports < 1:
        raise TouchstoneError(f"{path.name}: port count must be at least 1")

    freq_mult, kind, fmt, z0 = 1e9, "s", "ma", 50.0
    saw_option = False
    tokens: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.split("!", 1)[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                if not saw_option:
                    freq_mult, kind, fmt, z0 = _parse_option_line(line)
                    saw_option = True
                continue
            if line[0].isalpha():
                continue  # Touchstone 2.0 keyword block
            tokens.extend(line.replace(",", " ").split())

    try:
        numbers = np.asarray(tokens, dtype=float)
    except ValueError as exc:
        for tok in tokens:
            try:
                float(tok)
            except ValueError:
                raise TouchstoneError(
                    f"non-numeric token {tok!r} in {path.name}") from exc
        raise TouchstoneError(f"{path.name}: could not read the data ({exc})") from exc

    if numbers.size == 0:
        raise TouchstoneError(f"{path.name}: no data rows")
    if kind in ("h", "g"):
        raise TouchstoneError(
            f"{path.name}: {kind.upper()}-parameters are not supported. "
            "S, Y and Z are."
        )

    stride = 1 + 2 * n_ports * n_ports
    if numbers.size % stride:
        raise TouchstoneError(
            f"{path.name}: {numbers.size} numbers is not a multiple of {stride} "
            f"(1 frequency + {n_ports * n_ports} complex entries per row)."
        )

    rows = numbers.reshape(-1, stride)
    if not np.all(np.isfinite(rows)):
        n_bad = int(np.count_nonzero(~np.isfinite(rows)))
        raise TouchstoneError(
            f"{path.name}: {n_bad} non-finite value(s) (NaN/Inf). Refusing to "
            "parse -- anything computed from NaN is not a result."
        )

    freq = rows[:, 0] * freq_mult
    if np.any(np.diff(freq) <= 0):
        i = int(np.argmax(np.diff(freq) <= 0))
        raise TouchstoneError(
            f"{path.name}: frequencies are not strictly increasing at row {i + 2}"
        )

    pairs = rows[:, 1:].reshape(len(rows), n_ports * n_ports, 2)
    flat = _to_complex_array(pairs[:, :, 0], pairs[:, :, 1], fmt)
    if n_ports == 2:
        data = flat.reshape(len(rows), 2, 2).transpose(0, 2, 1)
    else:
        data = flat.reshape(len(rows), n_ports, n_ports)

    return Network(freq_hz=freq, s=np.ascontiguousarray(data), z0=z0,
                   kind=kind, path=str(path))


def write_touchstone(net: Network, path: str | Path, *, fmt: str = "ri",
                     freq_unit: str = "hz", precision: int = 12) -> Path:
    """Write a Network to a Touchstone 1.x file.

    Refuses to write anything it would refuse to read: a non-finite entry, a
    non-monotonic sweep, or an extension that disagrees with the port count.
    A writer that emits a file its own reader rejects is worse than no writer.
    """
    path = Path(path)
    fmt = fmt.lower()
    freq_unit = freq_unit.lower()
    if fmt not in FORMATS:
        raise TouchstoneError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    if freq_unit not in FREQ_UNITS:
        raise TouchstoneError(
            f"unknown frequency unit {freq_unit!r}; expected one of "
            f"{tuple(FREQ_UNITS)}")

    s = np.asarray(net.s)
    if s.ndim != 3 or s.shape[1] != s.shape[2]:
        raise TouchstoneError(f"expected (F, N, N) data, got shape {s.shape}")
    if not np.all(np.isfinite(s)):
        n_bad = int(np.count_nonzero(~np.isfinite(s)))
        raise TouchstoneError(
            f"refusing to write {n_bad} non-finite value(s): a file containing "
            "NaN or Inf is one every downstream tool will either reject or, "
            "worse, quietly believe."
        )
    if len(net.freq_hz) != len(s):
        raise TouchstoneError(
            f"{len(net.freq_hz)} frequencies but {len(s)} matrices")
    if np.any(np.diff(net.freq_hz) <= 0):
        raise TouchstoneError("frequencies must be strictly increasing")

    n = s.shape[1]
    m = re.search(r"\.s(\d+)p$", path.name, re.IGNORECASE)
    if m and int(m.group(1)) != n:
        raise TouchstoneError(
            f"filename says {m.group(1)} ports but the data has {n}. Touchstone "
            f"encodes the port count in the extension; use .s{n}p."
        )
    if not m:
        path = path.with_suffix(f".s{n}p")

    flat = s.transpose(0, 2, 1).reshape(len(s), -1) if n == 2 else s.reshape(len(s), -1)
    a, b = _from_complex_array(flat, fmt)
    scale = FREQ_UNITS[freq_unit]

    lines = [
        "! Written by touchstone-tools",
        f"! {n}-port {net.kind.upper()}-parameters, {len(s)} frequency points",
        f"# {freq_unit.upper()} {net.kind.upper()} {fmt.upper()} R {net.z0:g}",
    ]
    for i, f in enumerate(net.freq_hz):
        vals = " ".join(f"{x:.{precision}g} {y:.{precision}g}"
                        for x, y in zip(a[i], b[i]))
        lines.append(f"{f / scale:.{precision}g} {vals}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path

