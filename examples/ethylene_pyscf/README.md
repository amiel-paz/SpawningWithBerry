# Ethylene integration example

This SA(3)-CASSCF(2e,2o)/6-31G* closed-shell example exercises a larger PySCF
backend, the C-C and H-C-C-H observables, state gap/population output, and restart.
It is intentionally a slower integration workload rather than part of the fast test
suite.

`production.in` follows one initial condition for 10335 au (250 fs), matching the
trajectory duration used by Tao, Levine, and Martinez for their CASSCF/MS-CASPT2
ethylene comparison. It uses a ground-state harmonic Wigner sample and spin-pure,
density-fitted PySCF. The sourced production protocol is recorded in
`protocol-manifest.json`: ordinary/coupling nuclear steps are 20/5 au, with
transactional energy-gate refinement down to 0.625 au. Adaptive coefficient
substeps may refine to 0.00244140625 au without new PySCF calls.

All three singlet roots are included with equal state-average weights. The initial
state remains zero-based state 1 (S1); spawning may create children on S0 or S2.
The paper's CASSCF benchmark averaged 13 independently sampled initial conditions;
one production input is therefore one ensemble member, not a lifetime estimate.
`run_ensemble.py` runs the first 12 deterministic members in two concurrent waves
of six, using seeds 87062 through 87074 by default and resuming valid member
checkpoints. Each member uses one in-process PySCF evaluator and two numerical
threads. The final member receives the otherwise idle electronic slots. The runner
stops on a 32 GB aggregate RSS limit, sustained swap growth, a member failure, or a
projected makespan above 72 hours.

The production launch is deliberately gated:

```bash
# Six-member throughput and memory stress gate.
caffeinate -i .venv/bin/python examples/ethylene_pyscf/run_ensemble.py \
  --count 6 --parallel-members 6 --simulation-time 100 \
  --ensemble-name run-gate-100

# Two independent members through the known spawning region.
caffeinate -i .venv/bin/python examples/ethylene_pyscf/run_ensemble.py \
  --count 2 --parallel-members 2 --simulation-time 700 \
  --ensemble-name run-gate-700

# Only after the scientific and <72-hour projection gates pass.
caffeinate -i .venv/bin/python examples/ethylene_pyscf/run_ensemble.py
```

The rolling status, call counts, memory use, and makespan estimate are written to
`ensemble-status.json` in the selected ensemble directory.

### Recovery gates completed 2026-07-14

- The six-member 100-au no-renormalization stress gate completed in 12.38 s,
  peaked at 0.97 GB aggregate RSS, used no swap, and projected 0.77 h for 13
  members in the pre-spawn regime.
- A seed-87062 Verlet check gave maximum classical-energy drifts of
  `7.9849e-4`, `1.9219e-4`, and `4.8249e-5 Eh` at 20, 10, and 5 au. The factor
  of approximately four per halving confirms the expected second-order error.
- Two no-renormalization members completed 700 au. Seed 87062 spawned once at
  515 au and reached S0/S1 populations 0.667864/0.332136; seed 87063 did not
  enter a spawning region. The worst accumulated metric-norm and population-sum
  errors were `1.35e-13` and `9.13e-14`, and peak RSS was 0.56 GB.
- Repeating both members with `pair_overlap_threshold=1e-4` produced identical
  parentage, spawn time, and populations to stored precision. The production
  threshold therefore remains `1e-3`.
- The spawning gate projected a 13-member local makespan of 4.5--4.6 h from the
  observed workload. This is an early-trajectory estimate and is updated while
  production creates additional TBFs.
- A later seed-87063 800--820 au rejection exposed the apparent jump near 20 fs.
  The active pair rotated internally by about 47 degrees while its subspace stayed
  continuous (singular values 0.99894/0.99700). Comparing bare CI arrays therefore
  falsely suggested an S0/S1 swap; the old tracker then attached the S1 trajectory
  to the S0 energy and manufactured a 0.07626-Eh discontinuity. Root tracking now
  evaluates the many-electron overlap in the transformed old/new active-orbital
  bases. The corrected diagonal overlaps are 0.99427/0.99548/0.99720, energy order
  is unchanged, and the true S1 energy/gradient residual is only 5.28e-5 Eh. New
  checkpoints additionally Procrustes-align the active orbitals and transform the
  CI coefficients into that aligned representation, so subsequent guesses inherit
  a continuous full-CAS wavefunction gauge rather than PySCF's arbitrary one.
- After this gauge correction, all 43 tests, the six-member 100-au stress gate,
  and both two-member 700-au spawning gates were rerun from step zero. The two
  pair thresholds give bitwise-identical populations and spawn histories; worst
  root overlap is 0.97630, metric-norm error is 1.88e-13, quantum-energy drift is
  0.004673 Eh, and classical-energy drift is 7.98e-4 Eh. Fresh production was
  relaunched only after those gates passed.
- That launch subsequently exposed a localized 1.876-mEh Verlet defect for seed
  87066 at 1900 au, where the S0/S1 gap narrowed to 0.02425 Eh and the root overlap
  fell to 0.851. The error disappeared on the following interval, identifying
  finite-step quadrature rather than secular drift. The local electronic
  energy/gradient consistency gate is therefore 0.0001 Eh, separate from the
  0.005-Eh global diagnostic stop; intervals exceeding it transactionally retry
  from their starting checkpoint at 5 au.
- Driving an absolute energy wall with variable-step Verlet was also rejected:
  repeated 20/5/2.5-au transitions change the integrator's shadow Hamiltonian and
  can manufacture boundary-hugging offsets. Production therefore uses a fixed
  5-au velocity-Verlet nuclear step. This deliberately departs from the canonical
  20-au normal step to preserve a single symplectic map. The production hard bound
  is 0.2 mEh, while the local interval-consistency trigger remains 0.1 mEh. A
  separate 1e-8-Eh comparison margin reflects the CASSCF convergence floor; raw
  energies are stored unchanged and are never recentered.
- Spawn-child backpropagation uses the same local refinement
  ladder: seed 87063 demonstrated that a provisional child can cross the sharp
  995-au near-degeneracy even when its parent completed the forward interval.
  Rejected child trials are now discarded transactionally, leaving the threshold-
  entry replay snapshot and zero-amplitude insertion semantics unchanged.
- Quantum-energy excursions are stored rather than corrected. The exact seed-87063
  replay gives the same 5.83-mEh SPA0 Hamiltonian excursion at 0.625, 0.3125, and
  0.15625 au, while its metric norm remains converged. Production therefore uses
  `quantum_energy_policy record`: it never rescales coefficients, and all raw
  values remain available for analysis. Classical energy, norm, population sum,
  electronic continuity, and replay consistency retain hard failure gates.
- Centroid NAC phases are parallel-transported independently for every active TBF
  pair and included in exact replay checkpoints. This removes a scheduler-dependent
  sign reversal observed at 992.5 au: one- and six-worker replays now agree within
  `5.81e-4` in state population instead of differing by 0.222. The final from-zero
  six-worker 1100-au gate completed with S0 population 0.755661 at the old frontier,
  norm error `2.18e-13`, and maximum classical drift 0.0715 mEh.
- Final from-zero gates at fixed 5 au passed. Seed 87062's largest per-TBF
  classical drift through 700 au is 0.0673 mEh and it reproduces the 515-au spawn;
  seed 87063's drift is 0.0189 mEh. The `10^-3` and `10^-4` pair thresholds give
  identical frames, populations, parentage, and spawn decisions. Worst norm error
  is `1.90e-13`, worst root overlap is 0.97569, and the projected 13-member
  makespan is 6.35 hours.

The accepted equilibrium geometry is C=C 1.339 angstrom, C-H 1.086 angstrom, and
H-C-H 117.6 degrees. Its harmonic force field is a central finite-difference
MP2/6-31G* Hessian with the six external translation/rotation modes projected out;
the retained mass-weighted spectrum contains exactly 12 vibrational modes.

Production must start from step zero after changes to quantum propagation or spawning.
Do not reuse checkpoints from runs made before energy-reference-invariant generalized
Crank--Nicolson, adaptive metric compatibility, and transactional threshold-entry
quantum replay were introduced. Invalidated runs
are retained only as diagnostic archives under `run-production-invalid-*` and their
amplitudes/populations must not be analyzed as physical AIMS results.
