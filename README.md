# SpawningWithBerry (`aims-berry`)

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

Every run also maintains a human-readable mirror by default:

```text
run/readable/
├── populations.csv
├── quantum_diagnostics.csv
├── events.jsonl
├── spawns.csv
├── run_status.json
└── tbfs/
    ├── index.csv
    └── <label-id>/
        ├── energies.csv
        ├── phase_space.csv
        ├── couplings.csv
        └── derivative_norms.csv
```

These files contain committed frames only and update while propagation runs. HDF5
remains the authoritative full-array record. Existing histories can be exported
without rerunning dynamics using `aims-berry export-readable run/simulation.h5
--output run/readable`. Set `write_readable false` to disable the live mirror.

Check exactly which release and source revision is running:

```bash
aims-berry --version
aims-berry version --json
```

The detailed report includes the semantic package version, last-update timestamp,
Git commit and dirty-worktree status when run from a checkout, Python version, and
loaded package path. New HDF5 histories and checkpoints record the same provenance,
so a result remains traceable after it is copied to another machine.

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
The [corroboration report](docs/corroboration.md) records the quantitative comparison
with the PySpawn paper and separates published-data reproduction from new dynamics.
The [ethylene production report](docs/ETHYLENE_PRODUCTION.md) records the numerical
gates, timestep decision, spawn behavior, and final physical diagnostics.

## Connect an electronic-structure or MLIP engine

The dynamics driver does not import an electronic-structure package directly. It
sends an `ElectronicStructureRequest` to a provider and validates the returned
`ElectronicStructureResult`. Consequently, an in-house quantum-chemistry program,
an ML potential, a subprocess wrapper, or a remote service can drive AIMS without
changes to spawning, propagation, gauge tracking, storage, or analysis.

Each request supplies:

- `atoms` and `atomic_numbers`;
- `geometry` with shape `(natom, 3)` in bohr;
- zero-based `states`, the active state, and the simulation time in atomic units;
- the requested properties (`energies`, `gradients`, `nacs`, or `state_overlaps`);
- optional `gradient_states` and `nac_pairs` selections, plus an optional NAC gap
  cutoff, so an engine need not calculate unused derivatives;
- `previous`, an optional provider-owned wavefunction/model-state handle for guesses
  and root tracking.

The provider returns atomic-unit arrays with these conventions:

| Property | Shape | Convention |
| --- | --- | --- |
| Energies | `(nstate,)` | Real adiabatic state energies |
| Gradients | `(nstate, natom, 3)` | `dE/dR`, not negative forces |
| NACs | `(nstate, nstate, natom, 3)` | Complex `d[i,j] = <psi_i|grad psi_j>` |
| State overlaps | `(nstate, nstate)` | Complex `<psi_i(old)|psi_j(new)>` |

NACs must be anti-Hermitian in their state indices: `d[i,j] =
-conj(d[j,i])`. All returned values are checked for shape and finiteness before they
can enter the dynamics.

### Smallest Python adapter

Wrap a Python function with `CallableProvider`. This example represents the call to
`my_engine`; its API can be replaced by an ASE calculator, an MLIP library, a local
executable, or an RPC client.

```python
import numpy as np

from aims_berry import (
    CallableProvider,
    ElectronicStructureResult,
    ProviderCapabilities,
    load_config,
    run,
)


def calculate(request):
    raw = my_engine.evaluate(
        elements=request.atoms,
        atomic_numbers=request.atomic_numbers,
        positions_bohr=request.geometry,
        states=request.states,
        active_state=request.active_state,
        previous=request.previous,
    )

    # Convert the engine's native units and force convention here.
    energies = np.asarray(raw.energies_hartree)
    gradients = -np.asarray(raw.forces_hartree_per_bohr)
    nacs = np.asarray(raw.nacs_inverse_bohr, dtype=np.complex128)
    return ElectronicStructureResult(
        energies=energies,
        gradients=gradients,
        nacs=nacs,
        metadata={"engine": "my_engine", "model": raw.model_version},
    )


provider = CallableProvider(
    calculate,
    capabilities=ProviderCapabilities(
        energies=True,
        gradients=True,
        nacs=True,
        complex_values=True,
    ),
)
config = load_config("aims.in")
result = run(config, provider=provider)
print(result.history)
```

The callable may return the dictionary keys `energies`, `gradients`, `nacs`,
`state_overlaps`, `dipoles`, `charges`, `wavefunction`, and `metadata` instead of
constructing `ElectronicStructureResult` itself. See the runnable
[home-built calculator example](examples/home_baked/calculator.py).

### Load the adapter from the CLI

Put the adapter next to the input file as `engine_adapter.py`:

```python
from aims_berry import CallableProvider, ProviderCapabilities


def calculate(request):
    # Call the engine and return atomic-unit arrays with the shapes above.
    return engine_result(request)


def make_provider(config):
    return CallableProvider(
        calculate,
        capabilities=ProviderCapabilities(
            energies=True, gradients=True, nacs=True, complex_values=True
        ),
    )
```

Then select that factory in `aims.in`:

```text
provider custom
provider_option factory engine_adapter:make_provider

geometry molecule.xyz
geometry_units angstrom
num_states 2
initial_state 1
time_step 10 au
simulation_time 5000 au
initial_condition file

electronic_method custom
coupling_mode nac
spawn_threshold 0.01
population_to_spawn 0.001
spawn_cooldown 50
max_trajectories 32
run_directory run
```

Run and restart it using the same commands as a built-in backend:

```bash
aims-berry validate aims.in
aims-berry run aims.in
aims-berry restart run/checkpoint
```

### Using state overlaps instead of analytic NACs

An engine that produces reliable wavefunction overlaps between consecutive calls can
use overlap/NPI coupling. Return `state_overlaps`, retain enough information in
`WavefunctionState` to compare the next call with `request.previous`, and declare:

```python
ProviderCapabilities(
    energies=True,
    gradients=True,
    state_overlaps=True,
    npi_tdc=True,
    complex_values=True,
)
```

Set `coupling_mode npi`, or use `coupling_mode auto` to prefer certified overlaps
and fall back to analytic NACs when the provider supports only NACs. The driver
performs state assignment, phase alignment, and degenerate-subspace transport.

### Requirements for an MLIP

A conventional single-surface energy/force MLIP can run one-state Gaussian dynamics
with `num_states 1`, but it cannot produce nonadiabatic spawning by itself. A
multi-state AIMS run requires, at every requested geometry:

1. energies for every configured state and gradients for the states selected by
   `request.gradient_states` (normally only the active state); and
2. derivative couplings for `request.nac_pairs`, or certified consecutive-state
   overlaps when overlap/NPI coupling is selected.

Returning full gradient and NAC arrays remains backward compatible. Selective
providers return the standard full-sized arrays together with `gradient_mask` and
`nac_mask`; unavailable entries may be non-finite for gradients and zero for NACs,
and the core never indexes an unavailable derivative.

The MLIP must preserve a consistent state definition or provide overlap information
that lets the driver track roots. Uncertainty estimates, model versions, SCF flags,
and other diagnostics should be returned in `metadata`. An engine failure should
raise `ElectronicStructureError(..., retryable=True)` only when retrying the same
request can reasonably succeed; the driver never substitutes stale electronic data.

### Stateful engines and exact restart

For orbital guesses, neural-network hidden state, external checkpoint files, or a
remote job identifier, implement the full `ElectronicStructureProvider` protocol.
Subclassing `BaseProvider` provides no-op restart methods; override
`dump_state(directory)` and `load_state(directory, metadata)` to make provider state
part of the atomic simulation checkpoint. Only provider artifacts written inside the
given directory should be referenced by the returned JSON-serializable manifest.

The complete contract is in [the provider guide](docs/providers.md) and
[`src/aims_berry/electronic/base.py`](src/aims_berry/electronic/base.py).

## Scientific scope

The implementation includes analytic frozen-Gaussian matrix elements, an adaptive
metric-aware Crank--Nicolson propagator, velocity Verlet, Berry-curvature integration,
state/root tracking, phase and degenerate-subspace alignment, Wilson loops,
threshold/max-coupling spawning with energy-shell momentum adjustment, deterministic
task logging, append-only HDF5 history, and two-generation atomic checkpoints.

Quantum propagation uses the full right-acting moving-basis derivative in
`S c_dot + Sdot c = -i H c` and removes a scalar electronic-energy reference during
each rational step. Population transfer is therefore invariant to the absolute
electronic energy zero. The Hermitian part of the moving-basis derivative is made
discretely compatible with the endpoint overlap change; coefficient-only substeps
are doubled until both endpoint norm and phase-aligned coefficient gates pass,
without additional electronic-structure calls. When a coupling maximum is found, the driver restores its
threshold-entry snapshot, inserts the backpropagated child there with zero amplitude,
and replays the enlarged coupled basis to the previous frontier. It never inserts a
zero-amplitude child only after the coupling region has passed.

Replay history is transactional. Canonical HDF5 frames are visible only through the
last committed replay boundary; provisional frames live in a staging group and are
atomically committed at the previous frontier. An interrupted or failed replay
restarts from its exact threshold-entry state, queue, RNG state, gauge/provider
references, and staged-history window.

The H3 example is the fast SA-CASSCF benchmark used to generate bond/angle, gap, and
population datasets analogous to the observable classes in the PySpawn paper. The
ethylene example is intentionally marked as a slower integration workload. These are
benchmark workflows, not a claim to reproduce the paper's full 24-trajectory 4TCE
calculation.
