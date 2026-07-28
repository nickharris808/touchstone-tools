"""Every subcommand must survive a console that cannot encode the ohm sign.

All three of `info`, `convert` and `renorm` printed U+03A9 and therefore died
with ``UnicodeEncodeError`` on a default Windows console -- after doing the
work, so the file was written and the process still exited non-zero. A caller
reading the exit code would conclude the conversion was refused when it was
not. Found by running the published CLIs under `PYTHONIOENCODING=cp1252`
during final verification.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
LINE = HERE / "examples" / "line.s2p"
NARROW = ["cp1252", "ascii"]


def _run(args: list[str], encoding: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(HERE / "src"), PYTHONIOENCODING=encoding)
    return subprocess.run([sys.executable, "-m", "touchstone_tools.cli", *args],
                          cwd=HERE, capture_output=True, text=True, env=env)


@pytest.mark.parametrize("encoding", NARROW)
def test_info_survives_a_narrow_console(encoding: str) -> None:
    r = _run(["info", str(LINE)], encoding)
    assert "UnicodeEncodeError" not in r.stderr, r.stderr
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2-port S, 11 points" in r.stdout
    r.stdout.encode(encoding)


@pytest.mark.parametrize("encoding", NARROW)
def test_convert_survives_a_narrow_console(encoding: str, tmp_path: Path) -> None:
    out = tmp_path / "z.s2p"
    r = _run(["convert", str(LINE), str(out), "--to", "z"], encoding)
    assert "UnicodeEncodeError" not in r.stderr, r.stderr
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists(), "the file was written but the process reported failure"
    r.stdout.encode(encoding)


@pytest.mark.parametrize("encoding", NARROW)
def test_renorm_survives_a_narrow_console(encoding: str, tmp_path: Path) -> None:
    out = tmp_path / "r.s2p"
    r = _run(["renorm", str(LINE), str(out), "--z0", "75"], encoding)
    assert "UnicodeEncodeError" not in r.stderr, r.stderr
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists()
    r.stdout.encode(encoding)


def test_the_written_file_is_byte_identical_whatever_the_console_encoding(
    tmp_path: Path,
) -> None:
    """The glyph is a display concern. It must never reach the data."""
    a, b = tmp_path / "a.s2p", tmp_path / "b.s2p"
    _run(["renorm", str(LINE), str(a), "--z0", "75"], "utf-8")
    _run(["renorm", str(LINE), str(b), "--z0", "75"], "cp1252")
    assert a.read_bytes() == b.read_bytes()


def test_the_ohm_glyph_is_the_real_one_when_the_console_can_take_it() -> None:
    r = _run(["info", str(LINE)], "utf-8")
    assert "z0=50Ω" in r.stdout
