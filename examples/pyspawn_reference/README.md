# Published PySpawn reference-data reproduction

This workflow downloads the CC BY-NC 4.0 supporting archive for
DOI `10.1021/acs.jctc.0c00575.s001`, verifies its published MD5 checksum, and uses
the `aims_berry` analysis layer to regenerate the representative simulation's
Figure 6c distance, Figure 6d dihedral, Figure 7 energy gap, and coherent state
population datasets. It also compares the numerical summary with committed
reference values, so analysis drift fails loudly.

```bash
python fetch_and_analyze.py
```

The supporting archive is not committed. It is fetched from ACS Figshare file
`24165942`; generated data stays in the ignored `reference-output/` directory.
Figure 8 is a 24-simulation ensemble, while the published supporting archive contains
one representative simulation, so this workflow does not claim to regenerate the
Figure 8 ensemble average. The original PySpawn plotting helper converts Hartree to
the paper's eV-labelled scale with a factor of `13.6`, rather than the physical
`27.211386...`. Both datasets are exported: `figure7_gap.csv` reproduces the paper
convention and `figure7_gap_physical.csv` uses physical eV.

After fetching the history, an independent spin-pure PySCF calculation on its initial
4TCE geometry can be run with:

```bash
uv run --extra all python benchmark_pyscf.py reference-output/sim.hdf5
```

This SA(2)-CASSCF(2e,2o)/6-31G* point uses density fitting and takes roughly eight
minutes on a single-threaded macOS PySCF wheel. The committed result differs from the
raw TeraChem/PySpawn gap by only 0.00202 eV (0.05%).
