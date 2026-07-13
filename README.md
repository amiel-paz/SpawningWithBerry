# aims-berry

`aims-berry` is a modular, restartable implementation of trajectory-basis nonadiabatic
dynamics. Electronic structure is supplied through one small protocol, so PySCF,
analytic models, and private/home-built calculators all drive the same dynamics,
spawning, gauge, persistence, and analysis code.

The first physical backend is state-averaged CASSCF in PySCF, including analytic
state gradients and pairwise nonadiabatic couplings. The migrated two-dimensional
Berry model remains available for complex-coupling, curvature-force, Wilson-loop,
and gauge-invariance work.

## Install and run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[all,dev]'
aims-berry validate examples/h3_pyscf/aims.in
aims-berry run examples/h3_pyscf/aims.in
aims-berry analyze examples/h3_pyscf/run/simulation.h5 \
  --input examples/h3_pyscf/aims.in --output examples/h3_pyscf/analysis
```

Restart exactly from the most recent completed task:

```bash
aims-berry restart examples/h3_pyscf/run/checkpoint
```

Python callers use the same entry point:

```python
from aims_berry import load_config, run

result = run(load_config("examples/h3_pyscf/aims.in"))
```

The input is line-oriented: one snake-case keyword followed by space-delimited
values. `#` begins a comment and shell-style quoted paths are supported. Atomic units
and zero-based state/atom indices are the defaults. See
[the input reference](docs/input-reference.md), [provider guide](docs/providers.md),
and [legacy migration map](docs/migration.md).

## Scientific scope

The implementation includes analytic frozen-Gaussian matrix elements, a regularized
metric-aware Cayley propagator, velocity Verlet, Berry-curvature integration,
state/root tracking, phase and degenerate-subspace alignment, Wilson loops,
threshold/max-coupling spawning with energy-shell momentum adjustment, deterministic
task logging, append-only HDF5 history, and two-generation atomic checkpoints.

The H3 example is the fast SA-CASSCF benchmark used to generate bond/angle, gap, and
population datasets analogous to the observable classes in the PySpawn paper. The
ethylene example is intentionally marked as a slower integration workload. These are
benchmark workflows, not a claim to reproduce the paper's full 24-trajectory 4TCE
calculation.
