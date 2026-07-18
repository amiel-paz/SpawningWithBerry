# Deterministic PySpawn cone cross-engine benchmark

This reproduces `PySpawn/examples/test_cone/start.py` through both engines.  The
harmonized PySpawn input adds an inert third Cartesian component (mass `1e30`)
because the backend-neutral `ElectronicStructureRequest` uses molecular XYZ
geometry.  The physical x/y model, initial condition, widths, masses, timestep,
NPI overlap coupling, spawning threshold, and isotropic spawn momentum are
otherwise identical.

The benchmark runs pristine PySpawn/full diagonalization, harmonized
PySpawn/full diagonalization, and native SpawnWithBerry/Cayley.  No Cayley
adapter is used inside PySpawn. `compare.py` exports the numerical agreement
report after all runs finish.

Obtain PySpawn 1.7, apply its mechanical Python-3 conversion, and expose that
checkout as `PYSPAWN_ROOT`. Run native PySpawn from a clean directory with both
projects on `PYTHONPATH`:

```bash
export PYSPAWN_ROOT=/absolute/path/to/pyspawn17-py3
mkdir -p run-pyspawn-pristine-fulldiag
cd run-pyspawn-pristine-fulldiag
PYTHONPATH="$PYSPAWN_ROOT:../../../src" \
  ../../../.venv/bin/python "$PYSPAWN_ROOT/examples/test_cone/start.py"

cd ..
mkdir -p run-pyspawn-harmonized-fulldiag
cd run-pyspawn-harmonized-fulldiag
PYTHONPATH="$PYSPAWN_ROOT:../../../src" \
  ../../../.venv/bin/python ../pyspawn_start_harmonized.py
```

Run SpawnWithBerry from the example directory, then compare:

```bash
cd ..
../../.venv/bin/aims-berry run spawnwithberry.in
../../.venv/bin/python compare.py
```

The PySpawn tree requires only a mechanical Python-3 repair in
`pyspawn/potential/test_cone.py`: obsolete function-local `exec` assignments are
replaced by `getattr` calls.  The analytic model and `fulldiag` algorithm are
unchanged.
