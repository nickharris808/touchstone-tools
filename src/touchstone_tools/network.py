"""The Network container and the parameter conversions between S, Y and Z.

Conversions use the standard renormalized forms for a real reference impedance
z0, with the identity matrix scaled by z0:

    S = (Z - z0 I)(Z + z0 I)^-1
    Z = z0 (I + S)(I - S)^-1
    Y = Z^-1

Every one of these has a singular case, and each is **refused** rather than
regularized. A pure short has no admittance matrix; an ideal isolator has no
impedance matrix. Returning a large finite number there would be a confident
answer to a question with no answer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["Network", "TouchstoneError", "s_to_z", "z_to_s", "s_to_y", "y_to_s",
           "z_to_y", "y_to_z", "renormalize"]

#: Below this reciprocal condition number a matrix is treated as singular.
#: Chosen so that a matrix one part in 1e12 from singular is refused rather
#: than inverted into noise; float64 gives about 16 digits, so this leaves
#: four orders of margin over the arithmetic itself.
SINGULAR_RCOND = 1e-12


class TouchstoneError(ValueError):
    """Raised when a file cannot be parsed, or a conversion has no answer."""


@dataclass
class Network:
    """An N-port network sampled over frequency.

    Attributes
    ----------
    freq_hz : (F,) float, strictly increasing
    s       : (F, N, N) complex -- the parameter matrix, in whatever `kind` says
    z0      : reference impedance in ohms
    kind    : "s", "y" or "z"
    path    : source file, for messages
    """

    freq_hz: np.ndarray
    s: np.ndarray
    z0: float = 50.0
    kind: str = "s"
    path: str = "<memory>"

    @property
    def n_ports(self) -> int:
        return int(self.s.shape[1])

    @property
    def n_freq(self) -> int:
        return int(self.s.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"Network({self.n_ports}-port {self.kind.upper()}, {self.n_freq} freq, "
                f"{self.freq_hz[0]/1e9:.4g}-{self.freq_hz[-1]/1e9:.4g} GHz, z0={self.z0}Ω)")

    # ------------------------------------------------------------ conversions

    def to(self, kind: str) -> "Network":
        """Return this network as S, Y or Z parameters.

        Raises :class:`TouchstoneError` at any frequency where the conversion
        is singular, naming the frequency rather than returning a large number.
        """
        kind = kind.lower()
        if kind not in ("s", "y", "z"):
            raise TouchstoneError(f"unknown parameter kind {kind!r}; expected s, y or z")
        if kind == self.kind:
            return self
        fn = {
            ("s", "z"): s_to_z, ("s", "y"): s_to_y,
            ("z", "s"): z_to_s, ("z", "y"): z_to_y,
            ("y", "s"): y_to_s, ("y", "z"): y_to_z,
        }[(self.kind, kind)]
        return replace(self, s=fn(self.s, self.z0), kind=kind)

    def renormalized(self, z0: float) -> "Network":
        """Return the same network referenced to a different impedance."""
        return replace(self, s=renormalize(self.to("s").s, self.z0, z0),
                       z0=float(z0), kind="s")


def _inv_or_refuse(m: np.ndarray, what: str, freq_hz: np.ndarray) -> np.ndarray:
    """Batch-invert, refusing at any frequency where the matrix is singular."""
    # np.linalg.inv raises only on exact singularity; near-singular matrices
    # invert into numerical noise that looks like an answer. Check the
    # condition number instead and name the frequency that failed.
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.linalg.cond(m)
    bad = ~np.isfinite(cond) | (cond > 1.0 / SINGULAR_RCOND)
    if bad.any():
        i = int(np.argmax(bad))
        raise TouchstoneError(
            f"{what} is singular at {freq_hz[i] / 1e9:g} GHz "
            f"(condition number {cond[i]:.3e}). This conversion has no answer "
            "there -- a short has no admittance matrix and an ideal isolator "
            "has no impedance matrix -- so it is refused rather than "
            "regularized into a large finite number."
        )
    return np.linalg.inv(m)


def _eye_like(m: np.ndarray) -> np.ndarray:
    n = m.shape[-1]
    return np.broadcast_to(np.eye(n, dtype=complex), m.shape)


def s_to_z(s: np.ndarray, z0: float, freq_hz: np.ndarray | None = None) -> np.ndarray:
    """Z = z0 (I + S)(I - S)^-1."""
    f = np.arange(len(s)) if freq_hz is None else freq_hz
    i = _eye_like(s)
    return z0 * ((i + s) @ _inv_or_refuse(i - s, "(I - S)", f))


def z_to_s(z: np.ndarray, z0: float, freq_hz: np.ndarray | None = None) -> np.ndarray:
    """S = (Z - z0 I)(Z + z0 I)^-1."""
    f = np.arange(len(z)) if freq_hz is None else freq_hz
    i = _eye_like(z)
    return (z - z0 * i) @ _inv_or_refuse(z + z0 * i, "(Z + z0 I)", f)


def z_to_y(z: np.ndarray, z0: float = 0.0, freq_hz: np.ndarray | None = None) -> np.ndarray:
    """Y = Z^-1."""
    f = np.arange(len(z)) if freq_hz is None else freq_hz
    return _inv_or_refuse(z, "Z", f)


def y_to_z(y: np.ndarray, z0: float = 0.0, freq_hz: np.ndarray | None = None) -> np.ndarray:
    """Z = Y^-1."""
    f = np.arange(len(y)) if freq_hz is None else freq_hz
    return _inv_or_refuse(y, "Y", f)


def s_to_y(s: np.ndarray, z0: float, freq_hz: np.ndarray | None = None) -> np.ndarray:
    return z_to_y(s_to_z(s, z0, freq_hz), freq_hz=freq_hz)


def y_to_s(y: np.ndarray, z0: float, freq_hz: np.ndarray | None = None) -> np.ndarray:
    return z_to_s(y_to_z(y, freq_hz=freq_hz), z0, freq_hz)


def renormalize(s: np.ndarray, z_from: float, z_to_: float,
                freq_hz: np.ndarray | None = None) -> np.ndarray:
    """Re-reference an S-matrix from one real impedance to another.

    Done by going through Z, which is reference-independent, rather than with
    the direct S-to-S formula -- same result, and the singular case surfaces
    where it can be named.
    """
    if z_from <= 0 or z_to_ <= 0:
        raise TouchstoneError(
            f"reference impedance must be positive, got {z_from} -> {z_to_}"
        )
    return z_to_s(s_to_z(s, z_from, freq_hz), z_to_, freq_hz)
