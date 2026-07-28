"""Tests for touchstone-tools.

Two things carry most of the weight here. First, the 2-port column-major trap:
getting it wrong silently transposes every 2-port file, on read *and* on write,
and a round trip through a symmetric network would not notice. Second, the
refusals: a writer that emits what its reader rejects is worse than no writer.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
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
    renormalize,
    write_touchstone,
)
from touchstone_tools.cli import main as cli_main  # noqa: E402


def _net(n_ports=2, n_freq=8, z0=50.0, kind="s", asymmetric=False):
    f = np.linspace(1e9, 10e9, n_freq)
    rng = np.random.default_rng(0)
    s = 0.3 * (rng.standard_normal((n_freq, n_ports, n_ports))
               + 1j * rng.standard_normal((n_freq, n_ports, n_ports)))
    if not asymmetric:
        s = 0.5 * (s + s.transpose(0, 2, 1))
    return Network(freq_hz=f, s=s, z0=z0, kind=kind)


# ------------------------------------------------------------ round trips

@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("n_ports", [1, 2, 3, 4])
def test_write_then_read_is_a_round_trip(tmp_path, fmt, n_ports):
    net = _net(n_ports=n_ports)
    p = write_touchstone(net, tmp_path / f"x.s{n_ports}p", fmt=fmt)
    back = read_touchstone(p)
    assert back.n_ports == n_ports
    assert back.z0 == pytest.approx(net.z0)
    assert np.allclose(back.s, net.s, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("unit", ["hz", "khz", "mhz", "ghz"])
def test_frequency_units_round_trip(tmp_path, unit):
    net = _net()
    back = read_touchstone(write_touchstone(net, tmp_path / "u.s2p", freq_unit=unit))
    assert np.allclose(back.freq_hz, net.freq_hz, rtol=1e-9)


def test_two_port_column_major_survives_a_write_and_a_read(tmp_path):
    """S11 S21 S12 S22 on disk, [[S11,S12],[S21,S22]] in memory.

    An asymmetric network is essential here: a symmetric one round-trips
    perfectly through a transposing writer *and* a transposing reader, so the
    bug this pins would be invisible.
    """
    f = np.array([1e9])
    s = np.array([[[11 + 0j, 12 + 0j], [21 + 0j, 22 + 0j]]])
    p = write_touchstone(Network(freq_hz=f, s=s), tmp_path / "o.s2p", fmt="ri")

    numbers = [t for line in p.read_text().splitlines()
               if line and not line.startswith(("!", "#"))
               for t in line.split()]
    # frequency, then S11 S21 S12 S22 as real/imag pairs
    assert [float(numbers[i]) for i in (1, 3, 5, 7)] == [11.0, 21.0, 12.0, 22.0]

    back = read_touchstone(p)
    assert back.s[0, 0, 1].real == 12.0 and back.s[0, 1, 0].real == 21.0


def test_three_port_is_row_major(tmp_path):
    f = np.array([1e9])
    s = np.arange(1, 10, dtype=float).reshape(1, 3, 3).astype(complex)
    p = write_touchstone(Network(freq_hz=f, s=s), tmp_path / "o.s3p", fmt="ri")
    assert np.allclose(read_touchstone(p).s, s)


# ------------------------------------------------------------ conversions

@pytest.mark.parametrize("kind", ["z", "y"])
def test_s_converts_out_and_back(kind):
    net = _net()
    assert np.allclose(net.to(kind).to("s").s, net.s, rtol=1e-9, atol=1e-12)


def test_z_and_y_are_inverses():
    net = _net()
    z = net.to("z")
    assert np.allclose(z.to("y").to("z").s, z.s, rtol=1e-9, atol=1e-12)


def test_a_matched_load_has_the_expected_impedance():
    """S = 0 at every port means Z = z0 I. A concrete anchor for the algebra."""
    f = np.array([1e9, 2e9])
    net = Network(freq_hz=f, s=np.zeros((2, 2, 2), complex), z0=50.0)
    z = net.to("z").s
    assert np.allclose(z, 50.0 * np.eye(2), atol=1e-9)


def test_conversion_to_the_same_kind_is_the_identity():
    net = _net()
    assert net.to("s") is net


def test_unknown_kind_is_refused():
    with pytest.raises(TouchstoneError, match="unknown parameter kind"):
        _net().to("h")


# --------------------------------------------------------------- refusals

def test_a_singular_conversion_is_refused_not_regularized():
    """A perfect short has no admittance matrix. Say so; do not return 1e18.

    S = -I gives Z = 0, whose inverse does not exist. numpy would return
    values on the order of 1/eps and every one of them would look like a
    number a user could act on.
    """
    f = np.array([1e9, 2e9])
    s = -np.broadcast_to(np.eye(2, dtype=complex), (2, 2, 2)).copy()
    with pytest.raises(TouchstoneError) as exc:
        Network(freq_hz=f, s=s).to("y")
    assert "singular" in str(exc.value)
    assert "GHz" in str(exc.value), "the refusal must name the frequency"


def test_writing_nan_is_refused():
    net = _net()
    net.s[2, 0, 1] = np.nan
    with pytest.raises(TouchstoneError, match="non-finite"):
        write_touchstone(net, "unused.s2p")


def test_writing_a_backwards_sweep_is_refused(tmp_path):
    net = _net()
    net.freq_hz[3] = net.freq_hz[0]
    with pytest.raises(TouchstoneError, match="strictly increasing"):
        write_touchstone(net, tmp_path / "b.s2p")


def test_extension_must_agree_with_the_port_count(tmp_path):
    with pytest.raises(TouchstoneError, match=r"use \.s3p"):
        write_touchstone(_net(n_ports=3), tmp_path / "wrong.s2p")


def test_a_missing_extension_is_supplied(tmp_path):
    p = write_touchstone(_net(n_ports=4), tmp_path / "noext")
    assert p.name.endswith(".s4p")


def test_db_format_refuses_an_exact_zero(tmp_path):
    """0 is -inf dB: a correct value that cannot be written."""
    net = _net()
    net.s[0, 0, 0] = 0.0
    with pytest.raises(TouchstoneError, match="-inf dB"):
        write_touchstone(net, tmp_path / "z.s2p", fmt="db")


def test_reading_nan_is_refused(tmp_path):
    p = tmp_path / "n.s2p"
    p.write_text("# HZ S RI R 50\n1e9 nan 0 0.5 0 0.5 0 0.1 0\n")
    with pytest.raises(TouchstoneError, match="non-finite"):
        read_touchstone(p)


def test_h_parameters_are_refused_rather_than_misread(tmp_path):
    p = tmp_path / "h.s2p"
    p.write_text("# HZ H RI R 50\n1e9 1 0 2 0 3 0 4 0\n")
    with pytest.raises(TouchstoneError, match="H-parameters are not supported"):
        read_touchstone(p)


def test_bad_reference_impedance_is_refused():
    with pytest.raises(TouchstoneError, match="must be positive"):
        renormalize(_net().s, 50.0, -5.0)


def test_unknown_format_and_unit_are_refused(tmp_path):
    with pytest.raises(TouchstoneError, match="unknown format"):
        write_touchstone(_net(), tmp_path / "a.s2p", fmt="xy")
    with pytest.raises(TouchstoneError, match="unknown frequency unit"):
        write_touchstone(_net(), tmp_path / "a.s2p", freq_unit="furlongs")


# --------------------------------------------------------- renormalization

def test_renormalizing_there_and_back_is_the_identity():
    net = _net()
    assert np.allclose(net.renormalized(75.0).renormalized(50.0).s, net.s,
                       rtol=1e-9, atol=1e-12)


def test_renormalizing_a_matched_load_moves_it_off_match():
    """A 50-ohm matched load is not matched when referenced to 75 ohms."""
    f = np.array([1e9])
    net = Network(freq_hz=f, s=np.zeros((1, 1, 1), complex), z0=50.0)
    r = net.renormalized(75.0)
    expected = (50.0 - 75.0) / (50.0 + 75.0)      # reflection of 50 into 75
    assert r.s[0, 0, 0].real == pytest.approx(expected, abs=1e-12)


def test_renormalization_preserves_the_impedance_matrix():
    """Z is reference-independent; that is the whole reason it is the pivot."""
    net = _net()
    assert np.allclose(net.renormalized(75.0).to("z").s, net.to("z").s,
                       rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------- CLI

def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(argv)
    return rc, buf.getvalue()


def test_cli_info_reports_without_judging(tmp_path):
    p = write_touchstone(_net(asymmetric=True), tmp_path / "a.s2p")
    rc, out = _run(["info", str(p), "--json"])
    assert rc == 0
    d = json.loads(out)
    assert d["n_ports"] == 2 and d["kind"] == "s"
    assert d["max_abs_asymmetry"] > 0, "the fixture is asymmetric on purpose"
    # No verdict key: whether asymmetry is a defect is a question about the
    # device, and sparam-lint is the tool that answers it.
    assert not any(k in d for k in ("passed", "verdict", "ok"))


def test_cli_convert_and_renorm(tmp_path):
    src = write_touchstone(_net(), tmp_path / "s.s2p")
    rc, _ = _run(["convert", str(src), str(tmp_path / "z.s2p"), "--to", "z"])
    assert rc == 0
    assert read_touchstone(tmp_path / "z.s2p").kind == "z"

    rc, _ = _run(["renorm", str(src), str(tmp_path / "r.s2p"), "--z0", "75"])
    assert rc == 0
    assert read_touchstone(tmp_path / "r.s2p").z0 == pytest.approx(75.0)


def test_cli_refusal_exits_one_not_zero(tmp_path, capsys):
    """A refused conversion must not look like success."""
    f = np.array([1e9])
    s = -np.broadcast_to(np.eye(2, dtype=complex), (1, 2, 2)).copy()
    src = write_touchstone(Network(freq_hz=f, s=s), tmp_path / "short.s2p")
    rc = cli_main(["convert", str(src), str(tmp_path / "y.s2p"), "--to", "y"])
    assert rc == 1
    assert "refused" in capsys.readouterr().err


def test_cli_unreadable_file_exits_two(capsys):
    assert cli_main(["info", "/definitely/not/here.s2p"]) == 2
    assert "error" in capsys.readouterr().err


# ------------------------------------------- differential against sparam-lint

def test_reader_agrees_with_sparam_lint_when_it_is_installed(tmp_path):
    """Two independent readers of the same trap-laden format must agree.

    This is the highest-value test here: our reader and sparam-lint's were
    written separately, and the 2-port ordering is exactly the thing that
    would silently differ.
    """
    sl = pytest.importorskip("sparam_lint")
    net = _net(asymmetric=True)
    p = write_touchstone(net, tmp_path / "d.s2p", fmt="ri")
    theirs = sl.read_touchstone(p)
    ours = read_touchstone(p)
    assert ours.s.shape == theirs.s.shape
    assert np.allclose(ours.s, theirs.s, rtol=1e-12, atol=1e-15)
    assert ours.z0 == pytest.approx(theirs.z0)
    assert np.allclose(ours.freq_hz, theirs.freq_hz)


# ---------------------------------------------------------------- metadata

def test_version_matches_pyproject():
    import touchstone_tools
    text = (HERE / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{touchstone_tools.__version__}"' in text


def test_readme_quickstart_transcript_is_real_output(capsys):
    """The three commands in the card must print what the card says."""
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    ex = HERE / "examples" / "line.s2p"
    assert ex.exists(), "the quickstart references an example that is not shipped"

    rc, out = _run(["info", str(ex)])
    assert rc == 0
    for line in ("2-port S, 11 points, 1-10 GHz, z0=50Ω",
                 "max |entry|        0.9",
                 "max |S - S^T|      0.000e+00  (reported, not judged)"):
        assert line in out, f"info no longer prints: {line}"
        assert line in readme, f"the README no longer shows: {line}"


def test_example_file_regenerates_identically(tmp_path):
    """A shipped fixture nobody can rebuild is a fixture nobody can trust."""
    f = np.linspace(1e9, 10e9, 11)
    beta = 2 * np.pi * f / 3e8 * 0.05
    s = np.zeros((len(f), 2, 2), complex)
    s[:, 0, 0] = s[:, 1, 1] = 0.08 * np.exp(-2j * beta)
    s[:, 0, 1] = s[:, 1, 0] = 0.9 * np.exp(-1j * beta)
    fresh = write_touchstone(Network(freq_hz=f, s=s, z0=50.0),
                             tmp_path / "line.s2p", fmt="ri", freq_unit="ghz")
    assert (fresh.read_text(encoding="utf-8")
            == (HERE / "examples" / "line.s2p").read_text(encoding="utf-8"))


def test_readme_refusal_table_matches_the_code():
    """Every refusal the card promises must be one the code actually makes."""
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    from touchstone_tools.network import SINGULAR_RCOND
    assert f"SINGULAR_RCOND = {SINGULAR_RCOND:g}" in readme, (
        "the documented singularity threshold does not match the code"
    )
    for promised in ("NaN", "singular conversion", "strictly increasing",
                     "inf dB", "H- or G-parameters"):
        assert promised in readme, f"the refusal table no longer mentions: {promised}"

    # Where the card quotes an error message it must quote the real one. A
    # heading reasonably shortens at the comma, so compare that clause -- long
    # enough to catch a message that changed while the card did not.
    from touchstone_tools.io import _from_complex_array
    with pytest.raises(TouchstoneError) as exc:
        _from_complex_array(np.zeros(2, complex), "db")
    clause = str(exc.value).split(",")[0]
    assert clause in readme.replace("\u2212", "-"), (
        f"the card does not quote the real message: {clause!r}"
    )


def test_ci_installs_sparam_lint_so_the_differential_test_runs():
    """A skipped differential test is the one we least want to lose."""
    ci = (HERE / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "sparam-lint.git" in ci, (
        "CI no longer installs sparam-lint, so the cross-reader check will "
        "silently skip on every run"
    )
