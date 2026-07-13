# Input reference

Each nonempty line is `keyword value ...`; `#` begins a comment. Duplicate singleton
keywords and unknown keywords are errors. `provider_option` and `observable` may be
repeated. Paths are resolved relative to the input file.

Core controls are `provider`, `geometry`, `geometry_units`, `num_states`,
`initial_state`, `time_step`, `simulation_time`, `random_seed`, `initial_condition`,
`momenta`, `hessian`, and `temperature`. Times accept an optional `au` or `fs` suffix.

Electronic controls are `electronic_method`, `scf_method`, `basis`, `charge`, `spin`,
`active_electrons`, `active_orbitals`, `state_weights`, `coupling_mode`, and repeated
`provider_option name value` records. `coupling_mode auto` selects certified
overlap/NPI transport when available, otherwise analytic NACs.

`quantum_integrator cayley` is the regularized metric-aware default;
`quantum_integrator rk45` retains the research ODE path.

Spawning controls are `spawn_strategy`, `spawn_threshold`, `population_to_spawn`,
`spawn_overlap_max`, `spawn_cooldown`, `max_trajectories`, and `max_energy_gap`.
Numerical/storage controls include `regularization_threshold`, `overlap_threshold`,
`output_every`, `electronic_retries`, `checkpoint_keep`, and `run_directory`.

Observable records are zero-based:

```text
observable bond cc_distance 0 1
observable angle hch_angle 2 0 3
observable dihedral torsion 2 0 1 4
```

Coordinates are converted to bohr at input. Energies, gradients, momenta, masses,
NACs, propagation times, and all provider requests/results are in atomic units.
