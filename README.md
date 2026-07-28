# touchstone-tools

![CI](https://github.com/nickharris808/touchstone-tools/actions/workflows/ci.yml/badge.svg) ![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-blue) ![Licence](https://img.shields.io/badge/licence-Apache--2.0-green) ![Tests](https://img.shields.io/badge/tests-77%20passing-brightgreen)

📖 **[Documentation site](https://nickharris808.github.io/physics-lint/)** — the portfolio narrative, the concepts, a full walkthrough, and what all of this proves (and does not).

**Read, write and convert N-port network files — and refuse to emit one that
cannot be read back.**

```bash
pip install git+https://github.com/nickharris808/touchstone-tools.git

touchstone-tools info    measured.s2p
touchstone-tools convert measured.s2p out.s2p --to z
touchstone-tools renorm  measured.s2p out75.s2p --z0 75
```

> **Not yet on PyPI.** `pip install touchstone-tools` is the intended install
> once published; until then use the source install above.

## 30-second quickstart

The repository ships an 11-point lossy line at `examples/line.s2p`:

```bash
$ touchstone-tools info examples/line.s2p
line.s2p: 2-port S, 11 points, 1-10 GHz, z0=50Ω
  max |entry|        0.9
  max |S - S^T|      0.000e+00  (reported, not judged)

$ touchstone-tools convert examples/line.s2p line_z.s2p --to z
wrote line_z.s2p  (2-port Z, 11 points, z0=50Ω, RI)

$ touchstone-tools renorm examples/line.s2p line_75.s2p --z0 75
wrote line_75.s2p  (2-port S, 11 points, z0=75Ω, RI)
```

Note what `info` does **not** print. It reports the measured asymmetry
`max |S − Sᵀ|` and stops there. Whether that asymmetry is a defect is a question
about the *device* — a ferrite isolator is non-reciprocal by design — and
answering it is [`sparam-lint`](https://github.com/nickharris808/sparam-lint)'s
job, not this library's.

## Why this exists

`sparam-lint` tells you a model is broken. Nothing built one that cannot be.

Every Touchstone writer in the wild will happily emit a file with `NaN` in it, a
sweep that runs backwards, or a `.s2p` extension on 4-port data. Those files then
go into simulators. This library **refuses to write anything its own reader would
refuse to read** — which is a low bar that, as far as I can tell, nothing else
clears.

## The trap this format sets

Two-port files store data as **S11 S21 S12 S22** — column-major — while three-port
and above are row-major. A reader that assumes row-major everywhere silently
transposes every 2-port file it reads; a writer that does the same silently
transposes every one it emits.

Both directions are handled, and both are pinned by a test using four
distinguishable values `[[11, 12], [21, 22]]`. That test needs an **asymmetric**
network to mean anything: a symmetric one round-trips perfectly through a
transposing writer *and* a transposing reader, so the bug would be invisible.

There is also a differential test against `sparam-lint`'s independently written
reader — two separate implementations of the same trap-laden format, required to
agree to 1e-12.

## Library use

```python
from touchstone_tools import read_touchstone, write_touchstone

net = read_touchstone("measured.s2p")
print(net.n_ports, net.n_freq, net.z0, net.kind)

z = net.to("z")                       # S -> Z
y = net.to("y")                       # S -> Y
wide = net.renormalized(75.0)         # re-reference 50Ω -> 75Ω

write_touchstone(wide, "measured_75.s2p", fmt="ri", freq_unit="ghz")
```

| Object | What it is |
|---|---|
| `read_touchstone(path) -> Network` | Touchstone 1.x. Raises `TouchstoneError` on anything malformed. |
| `write_touchstone(net, path, fmt="ri", freq_unit="hz", precision=12)` | Writes and returns the path. Supplies the `.sNp` extension if absent. |
| `Network` | `.freq_hz` (F,), `.s` (F, N, N) complex, `.z0`, `.kind`, `.n_ports`, `.n_freq`, `.path` |
| `Network.to(kind)` | `"s"`, `"y"` or `"z"`. Returns self when already that kind. |
| `Network.renormalized(z0)` | Re-reference to a new real impedance. |
| `s_to_z`, `z_to_s`, `s_to_y`, `y_to_s`, `z_to_y`, `y_to_z` | The array-level conversions. |
| `renormalize(s, z_from, z_to)` | Array-level re-referencing. |
| `TouchstoneError` | `ValueError` subclass. Every refusal below raises it. |
| `FORMATS`, `FREQ_UNITS` | `("ri", "ma", "db")` and the Hz–THz multipliers. |

Renormalization goes **through Z**, which is reference-independent, rather than
using the direct S-to-S formula. Same answer, and the singular case surfaces
somewhere it can be named. A test asserts Z is unchanged by re-referencing,
which is the property that makes it the right pivot.

## What it refuses, and why

The rule: when there is no answer, say so. Never return a large finite number
that looks like one.

| Situation | What happens |
|---|---|
| `NaN`/`Inf` on read or write | Refused. Anything computed from `NaN` is not a result. |
| A singular conversion | Refused, **naming the frequency**. A perfect short has no admittance matrix; an ideal isolator has no impedance matrix. |
| Frequencies not strictly increasing | Refused on both read and write. |
| Extension disagrees with the port count | Refused, naming the extension that would fit. |
| Exactly zero written as `DB` | Refused: 0 is −inf dB, a correct value that cannot be written. Use `ri` or `ma`. |
| H- or G-parameters | Refused. S, Y and Z are supported; the rest are not, and guessing would be worse. |
| Reference impedance ≤ 0 | Refused. |

The singular case is the one worth dwelling on. `numpy.linalg.inv` raises only
on *exact* singularity; a matrix one part in 1e15 from singular inverts into
numerical noise, and every entry of that noise looks like a number you could act
on. So conversions check the condition number against `SINGULAR_RCOND = 1e-12`
and refuse above it, with the offending frequency in the message.

## CLI reference

```
touchstone-tools {info,convert,renorm} [...]
```

| Command | Arguments | What it does |
|---|---|---|
| `info` | `PATH [--json]` | Describe a file. Reports, never judges. |
| `convert` | `SRC DEST [--to {s,y,z}] [--format {ri,ma,db}] [--freq-unit ...]` | Convert kind and/or number format. |
| `renorm` | `SRC DEST --z0 OHMS [--format ...] [--freq-unit ...]` | Re-reference to a new impedance. |

| Code | Meaning |
|---|---|
| `0` | done |
| `1` | **refused** — the request has no answer, e.g. a singular conversion |
| `2` | usage, or a file that could not be read or written |

`1` and `2` are deliberately different. Exit 1 means the maths said no; exit 2
means the I/O did. Conflating them would hide which one happened.

## Troubleshooting

**`cannot infer port count from filename`** — Touchstone encodes it in the
extension. Rename to `.s2p`, `.s4p`, and so on.

**`filename says 2 ports but the data has 4`** — the writer will not paper over
this. Use the extension the message names.

**`Z is singular at 3 GHz`** — that conversion genuinely has no answer at that
frequency. If the network is a short, an open or an ideal isolator, this is
correct behaviour and not a bug to work around.

**`cannot write DB format: an entry is exactly zero`** — 0 is -inf dB. Write
`ri` or `ma`, both of which represent zero exactly.

**Values differ in the last digit after a round trip through `ma` or `db`** —
expected. `ri` is exact; the polar formats go through `exp`/`log` and agree to
about 1e-9 relative. Use `ri` when you need bit-exactness.

**`H-parameters are not supported`** — deliberate. Converting H to S needs
assumptions this library will not make for you.

## Scope, honestly

This is **Touchstone 1.x**. The 2.0 keyword block is skipped on read rather than
interpreted, so a v2 file with per-port reference impedances will be read with
the single `R` value from its option line — which may be wrong for that file.
That is a real limitation and the reason v2 is listed as future work rather than
claimed as supported.

Conversions assume a **real, scalar reference impedance**, which is what the
format's `R` field carries. Per-port and complex reference impedances are not
implemented.

Nothing here checks physics. A file this library is happy to write can still
describe a network that cannot exist; that is what `sparam-lint` is for.

## The rest of the toolkit

| | |
|---|---|
| [`physics-lint`](https://github.com/nickharris808/physics-lint) | One command for all the checkers. SARIF for CI. |
| [`sparam-lint`](https://github.com/nickharris808/sparam-lint) | Is an S-parameter model physically possible? Five laws + a negative control. |
| [`maxwell-lint`](https://github.com/nickharris808/maxwell-lint) | Does a coupling extractor predict impossible physics? Screening ceiling k ≤ 1. |
| [`abstain-bench`](https://github.com/nickharris808/abstain-bench) | Does a model know when to shut up? Abstention recall. |
| [`sparam-conformance`](https://huggingface.co/datasets/nickh007/sparam-conformance) | 11 labelled networks with verified ground truth. Grades the graders. |
| [`screening-ceiling`](https://huggingface.co/datasets/nickh007/screening-ceiling) | A certified impossibility result + 27 counterexamples. |
| [`physics-lint-action`](https://github.com/nickharris808/physics-lint-action) | The checks, in your CI. |
| [`physics-lint-mcp`](https://github.com/nickharris808/physics-lint-mcp) | A physics oracle your AI agent can call. |
| [**Try it in your browser**](https://huggingface.co/spaces/nickh007/physics-lint) | No install, runs client-side. |

These tools **grade** a model, or move one between formats. Producing one that is
passive *by construction* — so it cannot fail those laws whatever its parameters
— is the commercial core: **[ChipletOS](https://chipletos.com)**.

## Licence

Apache-2.0. See [LICENSE](LICENSE); copyright in [NOTICE](NOTICE).
