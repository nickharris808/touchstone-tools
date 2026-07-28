"""Adversarial tests for the read/write/convert surface.

Oracle: no input may produce a confident-looking answer that is wrong. This
library's specific temptation is that almost every refusal below has a
convenient alternative that returns a plausible number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))

from touchstone_tools import (  # noqa: E402
    FORMATS,
    Network,
    TouchstoneError,
    read_touchstone,
    write_touchstone,
)

_HDR = "# HZ S RI R 50"


def _net(n=2, f=6, asym=True):
    freq = np.linspace(1e9, 10e9, f)
    rng = np.random.default_rng(0)
    s = 0.3 * (rng.standard_normal((f, n, n)) + 1j * rng.standard_normal((f, n, n)))
    if not asym:
        s = 0.5 * (s + s.transpose(0, 2, 1))
    return Network(freq_hz=freq, s=s)


# ==================================================================== malformed

@pytest.mark.parametrize("content,label", [
    ("", "empty"),
    ("\n\n", "blank lines"),
    (_HDR + "\n", "no data rows"),
    ("! comment only\n", "comment only"),
    (_HDR + "\n1e9 0.1 0\n", "short row"),
    (_HDR + "\n1e9 0.1 0 0.5 0 0.5 0 0.1 0 EXTRA\n", "non-numeric trailing token"),
    (_HDR + "\n2e9 0.1 0 0.5 0 0.5 0 0.1 0\n1e9 0.1 0 0.5 0 0.5 0 0.1 0\n", "backwards"),
    (_HDR + "\n1e9 nan 0 0.5 0 0.5 0 0.1 0\n", "nan"),
    (_HDR + "\n1e9 inf 0 0.5 0 0.5 0 0.1 0\n", "inf"),
    ("# HZ H RI R 50\n1e9 1 0 2 0 3 0 4 0\n", "H-parameters"),
    ("# HZ S RI R notanumber\n1e9 1 0 2 0 3 0 4 0\n", "bad z0"),
])
def test_malformed_input_raises_rather_than_returning_a_network(tmp_path, content, label):
    p = tmp_path / "x.s2p"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(TouchstoneError):
        read_touchstone(p)


def test_a_directory_named_like_a_file_is_refused(tmp_path):
    (tmp_path / "trap.s2p").mkdir()
    with pytest.raises((TouchstoneError, IsADirectoryError, PermissionError)):
        read_touchstone(tmp_path / "trap.s2p")


def test_an_extension_with_no_port_count_is_refused(tmp_path):
    p = tmp_path / "model.txt"
    p.write_text(_HDR + "\n1e9 0.1 0 0.5 0 0.5 0 0.1 0\n", encoding="utf-8")
    with pytest.raises(TouchstoneError, match="port count"):
        read_touchstone(p)


def test_a_zero_port_extension_is_refused(tmp_path):
    p = tmp_path / "weird.s0p"
    p.write_text(_HDR + "\n1e9 0.1 0\n", encoding="utf-8")
    with pytest.raises(TouchstoneError):
        read_touchstone(p)


# ======================================================= write refuses the same

def test_the_writer_refuses_everything_the_reader_refuses(tmp_path):
    """The library's one rule, checked as a rule rather than case by case."""
    bad_nan = _net()
    bad_nan.s[1, 0, 0] = np.nan
    with pytest.raises(TouchstoneError, match="non-finite"):
        write_touchstone(bad_nan, tmp_path / "a.s2p")

    bad_inf = _net()
    bad_inf.s[1, 0, 0] = np.inf
    with pytest.raises(TouchstoneError, match="non-finite"):
        write_touchstone(bad_inf, tmp_path / "b.s2p")

    backwards = _net()
    backwards.freq_hz = backwards.freq_hz[::-1].copy()
    with pytest.raises(TouchstoneError, match="strictly increasing"):
        write_touchstone(backwards, tmp_path / "c.s2p")


def test_anything_written_can_be_read_back(tmp_path):
    """The property behind the rule, over every format and port count."""
    for n in (1, 2, 3, 4, 8):
        for fmt in FORMATS:
            net = _net(n=n, f=5)
            p = write_touchstone(net, tmp_path / f"n{n}_{fmt}.s{n}p", fmt=fmt)
            back = read_touchstone(p)
            assert back.n_ports == n
            assert np.allclose(back.s, net.s, rtol=1e-9, atol=1e-12), (
                f"{n}-port {fmt} did not survive a round trip"
            )


# ==================================================================== enormous

def test_a_large_sweep_round_trips_without_loss(tmp_path):
    net = _net(n=4, f=5000)
    back = read_touchstone(write_touchstone(net, tmp_path / "big.s4p"))
    assert back.n_freq == 5000
    assert np.allclose(back.s, net.s, rtol=1e-9, atol=1e-12)


def test_a_wide_port_count_round_trips(tmp_path):
    net = _net(n=16, f=20)
    back = read_touchstone(write_touchstone(net, tmp_path / "wide.s16p"))
    assert back.n_ports == 16
    assert np.allclose(back.s, net.s, rtol=1e-9, atol=1e-12)


# ========================================================= out of distribution

@pytest.mark.parametrize("scale", [1e-300, 1e-30, 1e30, 1e300])
def test_extreme_magnitudes_either_round_trip_or_refuse(tmp_path, scale):
    """No silent overflow into inf, and no silent flush to zero that lies."""
    net = _net(n=2, f=3)
    net.s = net.s * scale
    if not np.all(np.isfinite(net.s)):
        pytest.skip("the fixture itself overflowed; nothing to test")
    p = write_touchstone(net, tmp_path / "x.s2p", fmt="ri")
    back = read_touchstone(p)
    assert np.all(np.isfinite(back.s)), "reading back produced a non-finite value"
    assert np.allclose(back.s, net.s, rtol=1e-9, atol=0.0)


def test_a_single_frequency_point_is_fine():
    """One point is a legal sweep; nothing here takes a derivative."""
    net = Network(freq_hz=np.array([1e9]), s=np.zeros((1, 2, 2), complex))
    assert net.n_freq == 1
    assert np.allclose(net.to("z").s, 50.0 * np.eye(2), atol=1e-9)


def test_an_identity_s_matrix_has_no_impedance_and_says_so():
    """S = I is a perfect open: Z is infinite. Refuse, do not return 1e18."""
    f = np.array([1e9])
    s = np.broadcast_to(np.eye(2, dtype=complex), (1, 2, 2)).copy()
    with pytest.raises(TouchstoneError, match="singular"):
        Network(freq_hz=f, s=s).to("z")


def test_a_perfect_short_has_no_admittance_and_says_so():
    f = np.array([1e9])
    s = -np.broadcast_to(np.eye(2, dtype=complex), (1, 2, 2)).copy()
    with pytest.raises(TouchstoneError, match="singular"):
        Network(freq_hz=f, s=s).to("y")


def test_a_nearly_singular_conversion_is_refused_not_inverted_into_noise():
    """This is the case np.linalg.inv will happily answer, wrongly.

    inv() raises only on *exact* singularity. One part in 1e15 from singular it
    returns entries of order 1/eps, every one of which looks like a number a
    user could act on.
    """
    f = np.array([1e9])
    s = -np.broadcast_to(np.eye(2, dtype=complex), (1, 2, 2)).copy()
    s[0, 0, 0] += 1e-15
    with pytest.raises(TouchstoneError, match="singular"):
        Network(freq_hz=f, s=s).to("y")


# ================================================================= differential

def test_our_reader_agrees_with_sparam_lints(tmp_path):
    """Two independently written readers of the same trap-laden format."""
    sl = pytest.importorskip("sparam_lint")
    for n in (2, 3, 4):
        net = _net(n=n, f=8, asym=True)
        p = write_touchstone(net, tmp_path / f"d{n}.s{n}p", fmt="ri")
        ours, theirs = read_touchstone(p), sl.read_touchstone(p)
        assert np.allclose(ours.s, theirs.s, rtol=1e-12, atol=1e-15), (
            f"{n}-port: the two readers disagree"
        )
        assert np.allclose(ours.freq_hz, theirs.freq_hz)
        assert ours.z0 == pytest.approx(theirs.z0)


def test_both_readers_refuse_the_same_malformed_files(tmp_path):
    """Agreeing on what is unreadable matters as much as agreeing on values."""
    sl = pytest.importorskip("sparam_lint")
    for i, content in enumerate([
        _HDR + "\n1e9 nan 0 0.5 0 0.5 0 0.1 0\n",
        _HDR + "\n2e9 0.1 0 0.5 0 0.5 0 0.1 0\n1e9 0.1 0 0.5 0 0.5 0 0.1 0\n",
        _HDR + "\n1e9 0.1\n",
    ]):
        p = tmp_path / f"bad{i}.s2p"
        p.write_text(content, encoding="utf-8")
        with pytest.raises(TouchstoneError):
            read_touchstone(p)
        with pytest.raises(sl.TouchstoneError):
            sl.read_touchstone(p)


# ============================================== conversions are verdict-neutral

def test_a_conversion_round_trip_does_not_change_a_physics_verdict(tmp_path):
    """S -> Z -> S must not turn a passing model into a failing one.

    A format tool that quietly changes a verdict is worse than one that
    refuses: the number still looks right.
    """
    sl = pytest.importorskip("sparam_lint")
    net = _net(n=2, f=16, asym=False)
    net.s = net.s * 0.5      # comfortably passive

    p_before = write_touchstone(net, tmp_path / "before.s2p", fmt="ri")
    p_after = write_touchstone(net.to("z").to("s"), tmp_path / "after.s2p", fmt="ri")

    def verdicts(path):
        n = sl.read_touchstone(path)
        return {r.name: r.passed for r in sl.run_battery(n.s, n.freq_hz, n.z0)}

    assert verdicts(p_before) == verdicts(p_after)


def test_renormalizing_and_back_does_not_change_a_verdict(tmp_path):
    sl = pytest.importorskip("sparam_lint")
    net = _net(n=2, f=16, asym=False)
    net.s = net.s * 0.5

    p_before = write_touchstone(net, tmp_path / "b.s2p", fmt="ri")
    p_after = write_touchstone(net.renormalized(75.0).renormalized(50.0),
                               tmp_path / "a.s2p", fmt="ri")

    def verdicts(path):
        n = sl.read_touchstone(path)
        return {r.name: r.passed for r in sl.run_battery(n.s, n.freq_hz, n.z0)}

    assert verdicts(p_before) == verdicts(p_after)
