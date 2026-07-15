# Superseded ethylene diagnostics

> This report describes the earlier SA(2)-CASSCF(2e,2o) diagnostic. It is not the
> requested production model. Production now uses equal-weight
> SA(3)-CASSCF(2e,2o), starts on S1, and must begin again from time zero.
> The current corroboration targets are defined in `ETHYLENE_REFERENCE.md`.

The diagnostic trajectory uses seed 87062, SA(2)-CASSCF(2e,2o)/6-31G, a 5 au
outer nuclear step, and adaptive metric-aware coefficient propagation down to
0.00244140625 au. PySCF evaluations run in three isolated worker processes with
two OpenMP threads each. Process isolation is required because concurrent PySCF
calls in threads do not preserve deterministic gradients and NACs.

## Staged validation

- The 700 au dt=10 gate accepted spawns at 460 and 600 au and completed with
  three TBFs. Its largest raw endpoint norm drift was 7.59e-11, minimum gap was
  0.00199 Eh, quantum-energy drift was 0.00237 Eh, and the largest per-TBF
  classical-energy drift was 4.73e-4 Eh.
- The first spawn rejected the child's redundant back-spawn at threshold entry;
  the second same-state spawn completed with 1024 coefficient-only substeps.
- Independent dt=5 and dt=10 trajectories located both coupling maxima at 460
  and 600 au. Threshold entry times differed by only 5 au.
- dt=10 was rejected: its state population differed from dt=5 by 0.0779 at
  600 au and coefficient convergence missed the 1e-6 gate at 950 au even after
  4096 substeps.
- dt=5 completed 1000 au. Its largest raw norm drift was 9.27e-11, minimum gap
  was 0.00197 Eh, quantum-energy drift was 0.00261 Eh, and populations at
  1000 au were S0=0.976677 and S1=0.023323.

Those initial staged runs exposed a later SA-CASSCF orbital-minimum switch. The
provider now performs finite multi-start orbital following while retaining the
declared 2e2o active space: it solves from the transported active pair and from an
active/external exchange, then selects the converged stationary solution with the
largest active-subspace overlap. It records candidate energies, active-subspace
singular values, CI-root overlaps, root assignment, and energy/gradient consistency
residuals on every call. A one-step sequential/process-isolated comparison matched
all positions, momenta, energies, gradients, NACs, amplitudes, matrices, events,
queue state, and RNG state exactly.

## Production status

The final consistent 2e2o run is valid through 945 au. It reproduces accepted spawn
maxima at 460 and 600 au, rejects the redundant 460-au child back-spawn at threshold
entry, retains three TBFs, and reaches populations S0=0.97657 and S1=0.02343 by
800 au. Norm, population, overlap, Hermiticity, spawn-shell, and energy gates pass
through the last committed frame. At 945 au the populations are S0=0.976572 and
S1=0.023428. Across committed frames, the largest raw norm drift is 9.27e-11,
population-sum error is 9.53e-14, Hermiticity residual is zero to stored precision,
the minimum retained overlap eigenvalue is 0.99784, quantum-energy drift is
0.00261 Eh, and the largest per-TBF classical-energy drift is 1.01e-4 Eh. The
minimum state gap is 0.00197 Eh at 600 au. Valid active-subspace overlaps remain
above 0.997 before the rejected step.

The requested 5000-au calculation is **not complete**. Propagation from 945 to
950 au is rejected because the best converged 2e2o candidate has active-subspace
singular overlaps `[0.5446, 0.4564]`, below the documented 0.7 continuity gate.
Adding three external-orbital multi-start candidates does not improve that result.
This is a loss of active-space identity, not a CI-root phase/permutation problem.

SA-CASSCF(2e,3o) removes the discontinuity in the tested region, but it is retained
only as diagnostic evidence because it is not the same electronic Hamiltonian as
the requested and previously benchmarked 2e2o calculation. No tolerance was
loosened, no path-dependent energy correction was applied, and no 2e3o result is
reported as 2e2o production.

Valid-through-945-au numerical datasets and plots can be exported with
`aims-berry analyze examples/ethylene_pyscf/run-production --input
examples/ethylene_pyscf/production.in --output analysis`.

## Current SA(3) recovery protocol

None of the SA(2) numerical values above are production corroboration results. The
current calculation is equal-weight SA(3)-CAS(2e,2o)/6-31G* on S1, with 13
zero-temperature Wigner samples and 250 fs duration. Its complete sourced controls
and deviations are machine-readable in
`examples/ethylene_pyscf/protocol-manifest.json`.

The stopped seed-87062 SA(3) run is retained under
`run-ensemble-invalid-20260714-141318` as diagnostic-only data. It stopped during
the 490--545 au spawn replay at a 0.005081219 Eh quantum-energy drift. Direct
comparison of the canonical threshold-entry frame and the staged two-TBF insertion
frame proves that insertion preserved the metric norm and quantum energy exactly
and inserted the child with zero amplitude. The recovery therefore treats the
failure as a replay/nuclear-step propagation problem and retries the interval at
5, 2.5, 1.25, then 0.625 au while keeping canonical HDF5 history transactional.

No 13-member production run is launched until a six-member 100-au stress gate and
two independent 700-au spawning gates pass, and the measured local makespan is
below 72 hours.

Those recovery gates passed on 2026-07-14. The 700-au spawning member conserved
the unrenormalized metric norm to `1.35e-13`, inserted its child at exactly zero
amplitude, and reproduced the accepted 515-au spawn and substantial S1-to-S0
transfer at both pair thresholds. The 20/10/5-au Verlet energy-drift ratio is
consistent with second-order convergence. The measured early-trajectory makespan
projection is 4.5--4.6 hours for 13 members, with a 0.97 GB peak in the six-member
memory stress test. These values pass the launch gates but are not final 250-fs
scientific results.

The first full launch then exposed the apparent seed-87063 jump near 20 fs while
attempting 800--820 au. Its active orbital pair underwent an approximately
47-degree active-active rotation even though the physical active subspace remained
continuous (singular values `[0.99894, 0.99700]`). The provider had compared bare
CI coefficient arrays, which are expressed in different active-orbital bases. That
invalid comparison suggested assignment `[1,0,2]`, and the older implementation
permuted nondegenerate, energy-ordered roots. It thereby placed the S1 TBF on the S0
energy and manufactured a `-0.07625555 Eh` energy/gradient residual.

Root tracking now uses PySCF's determinant overlap with the complete old/new active-
orbital overlap transformation. On the archived endpoint the orbital-aware diagonal
root overlaps are `[0.994270, 0.995484, 0.997199]`, assignment remains `[0,1,2]`,
and the physical S1 energy/gradient residual is `-5.28123e-5 Eh`. Thus the large
jump was neither nuclear integration error nor a physical surface crossing. Bare-CI
overlaps are retained only as diagnostics. The persisted wavefunction is now put in
a parallel-transport gauge: a polar/Procrustes rotation maximally aligns the current
active orbitals with the previous ones, and every CI vector is transformed by the
corresponding determinant representation before root phase alignment. Genuine
energy/gradient or orbital-aware root-continuity rejections restore the complete
Verlet interval and follow the 5/2.5/1.25/0.625-au refinement ladder.

The corrected persistent gauge passed 43 tests and fresh from-zero launch gates.
The `10^-3` and `10^-4` 700-au pair-screening histories are identical: seed 87062
spawns at 515 au and reaches S0/S1 populations 0.667896/0.332104, while seed 87063
remains on S1. Across those gates the worst orbital-aware root overlap is 0.97630,
metric-norm error is `1.88e-13`, quantum-energy drift is `0.004673 Eh`, and maximum
per-TBF classical-energy drift is `7.98e-4 Eh`. The projected 13-member makespan is
3.34--3.40 hours. A new production ensemble was then launched from step zero; all
earlier runs containing bare-CI root tracking remain diagnostic-only.
