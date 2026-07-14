# Ethylene integration example

This SA(3)-CASSCF(2e,2o)/6-31G* closed-shell example exercises a larger PySCF
backend, the C-C and H-C-C-H observables, state gap/population output, and restart.
It is intentionally a slower integration workload rather than part of the fast test
suite.

`production.in` follows one initial condition for 10335 au (250 fs), matching the
trajectory duration used by Tao, Levine, and Martinez for their CASSCF/MS-CASPT2
ethylene comparison. It uses a ground-state harmonic Wigner sample and spin-pure,
density-fitted PySCF. A staged 10/5 au comparison chose
the 5 au outer step; adaptive quantum substeps may refine to 0.00244140625 au without
new PySCF evaluations. Its restartable
output is written to the ignored `run-production/` directory.

All three singlet roots are included with equal state-average weights. The initial
state remains zero-based state 1 (S1); spawning may create children on S0 or S2.
The paper's CASSCF benchmark averaged 13 independently sampled initial conditions;
one production input is therefore one ensemble member, not a lifetime estimate.

Production must start from step zero after changes to quantum propagation or spawning.
Do not reuse checkpoints from runs made before energy-reference-invariant generalized
Crank--Nicolson, adaptive metric compatibility, and transactional threshold-entry
quantum replay were introduced. Invalidated runs
are retained only as diagnostic archives under `run-production-invalid-*` and their
amplitudes/populations must not be analyzed as physical AIMS results.
