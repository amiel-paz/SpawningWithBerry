# PySpawn corroboration status

This repository distinguishes three different claims that are easy to conflate:

1. **Analysis reproduction:** `aims_berry` reads the public PySpawn supporting HDF5
   file and independently regenerates its observables.
2. **Backend integration:** the same driver runs real PySCF SA-CASSCF gradients and
   nonadiabatic couplings for H3 and ethylene.
3. **Independent dynamics reproduction:** a new 24-member, 4TCE AIMS ensemble with
   an equivalent electronic Hamiltonian. This has not been performed.

## Published supporting-data result

Run:

```bash
uv run --extra all python examples/pyspawn_reference/fetch_and_analyze.py
```

The script downloads ACS Figshare file `24165942`, verifies its published MD5
checksum, regenerates the plots/CSV files, and compares the summary against
`expected_metrics.json`.

The representative history contains one S1 parent and four S0 children. Its Figure
6 coordinate trends strongly corroborate the plotted behavior: the ring-closing
C-C distance falls from 5.2365 to about 3.68 Å by the first spawn, and the ethylenic
dihedral falls from 74.105° to about 36.54°. The children then separate into the
same shorter-bond and reopening branches visible in Figure 6. Spawn times are
164.97, 171.74, 197.62, and 204.15 fs. The coherent representative S1 population
falls to 0.0207, while the electronic-population norm is conserved to
`2.2e-11`.

## Discrepancies that prevent a stronger claim

- PySpawn's published plotting helper defines `au_to_ev = 13.6`. Consequently the
  Figure 7 vertical scale is half the physical Hartree-to-eV conversion. The adapter
  exports both the paper convention and physical eV instead of silently preserving
  the error.
- The public history reaches its minimum physical gap of 0.07794 eV at 165.45 fs.
  This does not agree with the paper text's statement that the near-zero-gap region
  is entered at 125 fs. The same history's first spawn at 164.97 fs agrees with the
  branch onset visible in Figure 6.
- The exact public-history values at first spawn (3.6805 Å and 36.54°) differ from
  the prose values (3.9 Å and 44°), although the regenerated curves match the Figure
  6 curves.
- The supporting archive contains only the representative simulation. It cannot
  reproduce Figure 8's 24 individual population traces or ensemble mean.

## PySCF checks

The H3 example completes a short SA-CASSCF(3e,3o)/STO-3G run and exercises spawning.
The ethylene example completes and restarts a five-step
SA(2)-CASSCF(2e,2o)/6-31G run with analytic gradients and NACs. Its duration is only
1 atomic unit (0.024 fs), so its constant S1 population is an integration check, not
evidence for the 4TCE photodynamics.

An independent spin-pure PySCF SA(2)-CASSCF(2e,2o)/6-31G* calculation was also run
on the public history's initial 36-atom 4TCE geometry. It gives a physical S1-S0 gap
of 4.014213 eV, versus 4.016229 eV in the TeraChem/PySpawn history: a difference of
0.002017 eV, or 0.0502%. Both roots have measured S-squared below `1e-20`. This
corroborates the raw electronic gap and independently supports the diagnosis that
Figure 7's plotted eV scale is low by a factor of two.

During this check, the provider was hardened to spin-purify every SA-CASSCF root by
default. Before that correction, an unconstrained two-root calculation could select
a triplet as the second root. The ethylene run was repeated after the correction;
its two singlet roots have S-squared approximately zero and its corrected initial
gap is 0.386881 Hartree.

The defensible conclusion is therefore: Figure 6 and the representative population
transfer are corroborated by the published numerical history; the raw initial 4TCE
gap is independently reproduced with PySCF; Figure 7 exposes a unit and timing
inconsistency; Figure 8 cannot be independently checked from the released archive;
and a fresh PySCF 4TCE dynamics ensemble remains future work.
