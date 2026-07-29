# Input reference

Each nonempty line is `keyword value ...`; `#` begins a comment. Duplicate singleton
keywords and unknown keywords are errors. `provider_option` and `observable` may be
repeated. Paths are resolved relative to the input file.

Core controls are `provider`, `geometry`, `geometry_units`, `num_states`,
`initial_state`, `time_step`, `simulation_time`, `random_seed`, `initial_condition`,
`momenta`, `hessian`, and `temperature`. Times accept an optional `au` or `fs` suffix.
`time_step` is the ordinary nuclear step. `coupling_time_step` is used in an active
spawning region, and `minimum_nuclear_time_step` is the hard floor for transactional
energy-gate retries. `min_time_step` is separate: it controls coefficient-only
substeps and never causes an extra electronic-structure call.

Electronic controls are `electronic_method`, `scf_method`, `basis`, `charge`, `spin`,
`active_electrons`, `active_orbitals`, `state_weights`, `coupling_mode`, and repeated
`provider_option name value` records. `coupling_mode auto` selects certified
overlap/NPI transport when available, otherwise analytic NACs.

`quantum_integrator cayley` is the regularized metric-aware default;
`quantum_integrator rk45` retains the research ODE path.
The Cayley integrator gates the raw endpoint metric norm and stores it unchanged.
Accepted coefficients are never rescaled. The adaptive map takes the endpoint-metric
polar factor of the linear Cayley propagator, enforcing the TDSE's metric-unitary
structure for every coefficient vector rather than normalizing a particular state.
A material cumulative norm drift remains a hard failure.

Spawning controls are `spawn_strategy`, `spawn_momentum`, `spawn_threshold`,
`population_to_spawn`, `spawn_metric`, `spawn_overlap_max`, `spawn_cooldown`,
`max_trajectories`, and `max_energy_gap`. `spawn_metric` may be `projected`,
`nac_norm`, or `tdc`; the latter is the absolute provider-certified NPI
time-derivative coupling. `spawn_momentum isotropic` reproduces PySpawn's
direction-preserving energy-shell rescaling for overlap-only providers;
`spawn_momentum nac` is the normal analytic-NAC adjustment. `nac_gap_threshold`
suppresses requested NAC pairs above the configured energy gap.
Numerical/storage controls include `regularization_threshold`, `overlap_threshold`,
`pair_overlap_threshold`, `energy_tolerance`, `quantum_energy_policy`,
`classical_energy_tolerance`,
`output_every`, `write_xyz`, `electronic_retries`, `checkpoint_keep`, and
`run_directory`. `write_xyz true` writes one Angstrom XYZ file per TBF under
`run_directory/geometries/step-<frame>/` whenever a committed HDF5 frame is
written; its cadence therefore follows `output_every`. Replay-staged frames are
not exported.
`classical_energy_tolerance` optionally imposes a per-TBF energy-drift hard stop;
when omitted it inherits `energy_tolerance`. Local energy/gradient work consistency,
electronic continuity, and coupling-region entry drive transactional nuclear-step
refinement. The global energy gate deliberately does not switch between Verlet
maps. `classical_energy_numerical_margin` is an explicit
comparison-only allowance for electronic convergence noise; it never changes or
recenters the recorded raw energy. TBF pairs whose analytic nuclear overlap is below
`pair_overlap_threshold` receive exactly zero off-diagonal matrix elements and do
not issue a centroid electronic-structure request.

`energy_tolerance` marks quantum-energy excursions for diagnostics.
`quantum_energy_policy record` (the default) preserves the raw TDSE solution and
records threshold crossings; `error` makes the same threshold a hard stop.  This
policy never rescales amplitudes. Metric norm and population-sum errors remain hard
failures regardless of the energy policy.

Providers may honor selective derivative requests. `gradient_states` and
`nac_pairs` are carried by the Python `ElectronicStructureRequest`, not by input
keywords: the driver asks for one active-state gradient at each TBF, no derivative
at same-state centroids, and at most the relevant state pair at an interstate
centroid. Legacy providers that return all derivatives remain valid.

Observable records are zero-based:

```text
observable bond cc_distance 0 1
observable angle hch_angle 2 0 3
observable dihedral torsion 2 0 1 4
```

Coordinates are converted to bohr at input. Energies, gradients, momenta, masses,
NACs, propagation times, and all provider requests/results are in atomic units.
`gaussian_widths` and `nuclear_masses` accept either one value per atom or three
Cartesian values per atom; explicit masses are principally useful for analytic
models and exact cross-engine fixtures.
