# Contributing to touchstone-tools

Thanks for looking. This project has one unusual rule, and it is the important one.

## The writer may never emit what the reader would refuse

Every guard on the read path needs a matching guard on the write path, and the
matching test. If `read_touchstone` rejects a file containing `NaN`, then
`write_touchstone` must refuse to produce one — otherwise this library becomes a
source of exactly the files it exists to keep out of simulators.

Concretely, a change that adds a refusal needs three things:

1. the refusal itself, raising `TouchstoneError` with a message that says what
   to do rather than only what went wrong;
2. a test that the refusal happens;
3. a check that the *other* direction agrees. A reader guard without a writer
   guard is a half-built fence.

## Never regularize a singular answer

When a conversion has no answer — a short has no admittance matrix, an ideal
isolator has no impedance matrix — **refuse and name the frequency**. Do not
return a large finite number.

This is easy to get wrong by accident, because `numpy.linalg.inv` raises only on
*exact* singularity. A matrix one part in 1e15 from singular inverts into
numerical noise, and every entry of that noise looks like a value a user could
act on. That is why conversions check a condition number rather than catching an
exception.

If you are tempted to add a `regularize=True` option: the answer is no. A caller
who wants a pseudo-inverse can compute one, and will know that they did.

## Report, do not judge

`info` prints `max |S − Sᵀ|` and stops. Whether that asymmetry is a *defect* is a
question about the device — a ferrite isolator is non-reciprocal by design — and
[`sparam-lint`](https://github.com/nickharris808/sparam-lint) is the tool that
answers it. Please do not add verdict fields here; the split between the two
libraries is deliberate.

## The 2-port trap

Two-port files are **column-major** (S11 S21 S12 S22); three-port and above are
row-major. Any change touching the reshape needs an **asymmetric** fixture,
because a symmetric network round-trips perfectly through a transposing writer
*and* a transposing reader — the bug survives every test built on a symmetric
network.

## Running the tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

The differential test against `sparam-lint`'s independently written reader skips
if that package is absent. It is the most valuable test here — two separate
implementations of the same trap-laden format, required to agree — so please
install it before submitting:

```bash
pip install git+https://github.com/nickharris808/sparam-lint.git@main
```

## Style

`ruff check .` must pass. Beyond that: comments should explain *why*, especially
where the code refuses to do something a reasonable person would expect it to do.
Those are the places a future contributor will otherwise "fix".
