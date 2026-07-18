"""PySCF SA-CASSCF electronic-structure provider."""

from __future__ import annotations

import json
import uuid
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import (
    BaseProvider,
    ElectronicProperties,
    ElectronicStructureError,
    ElectronicStructureRequest,
    ElectronicStructureResult,
    ProviderCapabilities,
    WavefunctionState,
)


def _energy_ordered_root_phases(
    tracking_overlap: np.ndarray,
    minimum_overlap: float,
    *,
    enforce: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase energy-ordered roots and diagnose any tempting diabatic reassignment.

    The Hungarian assignment is diagnostic only.  Permuting nondegenerate CASSCF
    roots would change the physical adiabatic surface associated with a state index.
    """
    overlap = np.asarray(tracking_overlap, dtype=np.complex128)
    row, col = linear_sum_assignment(-np.abs(overlap))
    assignment = col[np.argsort(row)]
    diagonal = np.diag(overlap)
    magnitudes = np.abs(diagonal)
    if enforce and magnitudes.size and float(np.min(magnitudes)) < minimum_overlap:
        raise ElectronicStructureError(
            "PySCF CI-root continuity failure: "
            f"diagonal_overlaps={magnitudes.tolist()}, "
            f"assignment_suggestion={assignment.tolist()}, "
            f"minimum={minimum_overlap}",
            retryable=False,
        )
    phases = np.exp(
        -1j * np.angle(np.where(magnitudes > 1.0e-14, diagonal, 1.0))
    )
    return phases, assignment, magnitudes


class PySCFProvider(BaseProvider):
    capabilities = ProviderCapabilities(energies=True, gradients=True, nacs=True)

    def __init__(
        self,
        *,
        basis: str,
        charge: int,
        spin: int,
        scf_method: str,
        active_electrons: int,
        active_orbitals: int,
        state_weights: tuple[float, ...],
        options: dict[str, Any] | None = None,
    ) -> None:
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.scf_method = scf_method.lower()
        self.active_electrons = active_electrons
        self.active_orbitals = active_orbitals
        self.state_weights = state_weights
        self.options = options or {}
        continuity_policy = str(
            self.options.get("continuity_policy", "error")
        ).lower()
        if continuity_policy not in {"error", "record"}:
            raise ElectronicStructureError(
                "PySCF continuity_policy must be 'error' or 'record'",
                retryable=False,
            )
        self._states: dict[str, dict[str, Any]] = {}
        self._checkpoint_references: frozenset[str] | None = None

    def set_checkpoint_references(self, identifiers: frozenset[str]) -> None:
        """Retain only wavefunctions needed by live TBFs, centroids, or spawn replay."""
        self._checkpoint_references = frozenset(identifiers)
        self._states = {
            identifier: state
            for identifier, state in self._states.items()
            if identifier in self._checkpoint_references
        }

    def adopt_wavefunction(self, wavefunction: WavefunctionState | None) -> None:
        """Register a result evaluated by an isolated electronic worker."""

        if wavefunction is not None and isinstance(wavefunction.payload, dict):
            self._states[wavefunction.identifier] = wavefunction.payload

    def _imports(self):
        try:
            from pyscf import fci, gto, mcscf, scf
        except ImportError as exc:
            raise ElectronicStructureError("PySCF provider requires `pip install aims-berry[pyscf]`") from exc
        return fci, gto, mcscf, scf

    def _previous_payload(self, previous: WavefunctionState | None) -> dict[str, Any] | None:
        if previous is None:
            return None
        if isinstance(previous.payload, dict):
            return previous.payload
        return self._states.get(previous.identifier)

    @staticmethod
    def _rohf_nac_context(casscf):
        """Work around PySCF's ROHF core-density shape mismatch in SA-CASSCF NACs.

        PySCF 2.13's NAC helper installs a spin-summed 2-D energy-weighted density
        into the ROHF gradient, whose UHF-derived implementation expects alpha/beta
        components. Supplying two half-density components is algebraically identical.
        """
        if casscf.mol.spin == 0:
            return nullcontext()
        from pyscf import lib
        from pyscf.nac import sacasscf as nac_module

        def rohf_core(mc_grad, mo_coeff=None, atmlst=None, eris=None, mf_grad=None):
            mc = mc_grad.base
            mo_coeff = mc.mo_coeff if mo_coeff is None else mo_coeff
            eris = mc.ao2mo(mo_coeff) if eris is None else eris
            mf_grad = mc._scf.nuc_grad_method() if mf_grad is None else mf_grad
            ncore = mc.ncore
            mo_h = mo_coeff.conj().T
            fock = mo_h @ mc.get_hcore() @ mo_coeff + eris.vhf_c
            mo_energy = fock.diagonal().copy()
            mo_occ = np.zeros_like(mo_energy)
            mo_occ[:ncore] = 2.0
            fock *= mo_occ[None, :]

            def spin_energy_density(*_args):
                summed = mo_coeff @ ((fock + fock.T) * 0.5) @ mo_h
                return np.asarray((0.5 * summed, 0.5 * summed))

            with lib.temporary_env(mf_grad, make_rdm1e=spin_energy_density, verbose=0):
                with lib.temporary_env(mf_grad.base, mo_coeff=mo_coeff, mo_occ=mo_occ):
                    return mf_grad.grad_elec(
                        mo_coeff=mo_coeff, mo_energy=mo_energy, mo_occ=mo_occ, atmlst=atmlst
                    )

        return lib.temporary_env(nac_module, grad_elec_core=rohf_core)

    def evaluate(self, request: ElectronicStructureRequest) -> ElectronicStructureResult:
        started = time.perf_counter()
        timings: dict[str, Any] = {}
        self.capabilities.require(request.properties)
        fci, gto, mcscf, scf = self._imports()
        molecule = gto.M(
            atom=[(atom, tuple(position)) for atom, position in zip(request.atoms, request.geometry)],
            unit="Bohr",
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
            symmetry=False,
            verbose=int(self.options.get("verbose", 0)),
            max_memory=float(self.options.get("max_memory", 4000)),
        )
        method = {"rhf": scf.RHF, "rohf": scf.ROHF, "uhf": scf.UHF}.get(self.scf_method)
        if method is None:
            raise ElectronicStructureError(f"unsupported scf_method {self.scf_method!r}")
        mean_field = method(molecule)
        if bool(self.options.get("density_fit", False)):
            auxiliary_basis = self.options.get("density_fit_auxbasis")
            mean_field = mean_field.density_fit(auxbasis=auxiliary_basis)
        mean_field.conv_tol = float(self.options.get("scf_conv_tol", 1.0e-10))
        previous = self._previous_payload(request.previous)
        scf_initialization = "rhf_bootstrap"
        initial_ci = None
        if (
            previous is not None
            and previous.get("mo_coeff") is not None
            and previous.get("molecule") is not None
            and self.scf_method != "uhf"
        ):
            # RHF/ROHF is only a bootstrap for the first point on an electronic
            # path.  At later geometries, project the *converged CASSCF* orbital
            # frame into the new AO basis, symmetrically orthonormalize it, and
            # enter the next SA-CASSCF macroiteration directly with the prior CI
            # vectors.  Re-solving HF here discards precisely the multireference
            # continuation information that dynamics needs.
            phase_started = time.perf_counter()
            try:
                projected = scf.addons.project_mo_nr2nr(
                    previous["molecule"],
                    np.asarray(previous["mo_coeff"]),
                    molecule,
                )
                overlap = mean_field.get_ovlp()
                metric = projected.conj().T @ overlap @ projected
                eigenvalues, eigenvectors = np.linalg.eigh(metric)
                if eigenvalues.size == 0 or float(np.min(eigenvalues)) < 1.0e-10:
                    raise ValueError(
                        "projected CASSCF orbital frame is linearly dependent"
                    )
                inverse_sqrt = (
                    eigenvectors
                    @ np.diag(eigenvalues ** -0.5)
                    @ eigenvectors.conj().T
                )
                initial_mos = np.real_if_close(projected @ inverse_sqrt)
                mean_field.mo_coeff = initial_mos
                mean_field.mo_energy = np.real_if_close(np.einsum(
                    "pi,pq,qi->i",
                    initial_mos.conj(),
                    mean_field.get_hcore(),
                    initial_mos,
                ))
                mean_field.mo_occ = mean_field.get_occ(
                    mean_field.mo_energy, initial_mos
                )
                mean_field.e_tot = mean_field.energy_tot(
                    dm=mean_field.make_rdm1(initial_mos, mean_field.mo_occ)
                )
                initial_ci = previous.get("ci")
            except Exception as exc:
                raise ElectronicStructureError(
                    f"PySCF CASSCF orbital transport failed: {exc}",
                    retryable=False,
                ) from exc
            timings["orbital_transport_seconds"] = (
                time.perf_counter() - phase_started
            )
            timings["scf_seconds"] = 0.0
            scf_initialization = "casscf_transport"
        else:
            phase_started = time.perf_counter()
            mean_field.kernel()
            timings["scf_seconds"] = time.perf_counter() - phase_started
            if not mean_field.converged:
                raise ElectronicStructureError(
                    "PySCF SCF did not converge", retryable=True
                )
            initial_mos = np.asarray(mean_field.mo_coeff)
        target_spin = 0.5 * molecule.spin
        target_spin_square = float(
            self.options.get("spin_square", target_spin * (target_spin + 1.0))
        )
        weights = self.state_weights or tuple([1.0 / len(request.states)] * len(request.states))

        def configured_casscf():
            solver = mcscf.CASSCF(
                mean_field, self.active_orbitals, self.active_electrons
            )
            solver.conv_tol = float(self.options.get("casscf_conv_tol", 1.0e-8))
            solver.max_cycle_macro = int(self.options.get("casscf_max_cycle", 50))
            solver.max_stepsize = float(
                self.options.get("casscf_max_stepsize", solver.max_stepsize)
            )
            if bool(self.options.get("fix_spin", True)):
                solver.fix_spin_(
                    shift=float(self.options.get("spin_penalty", 0.5)),
                    ss=target_spin_square,
                )
            return solver.state_average(weights)

        template = configured_casscf()
        if (
            scf_initialization == "rhf_bootstrap"
            and previous is not None
            and previous.get("mo_coeff") is not None
        ):
            try:
                initial_mos = mcscf.addons.project_init_guess(
                    template, previous["mo_coeff"], prev_mol=previous.get("molecule")
                )
                initial_ci = previous.get("ci")
            except Exception:
                initial_mos = np.asarray(mean_field.mo_coeff)
                initial_ci = None
        guesses: list[tuple[np.ndarray, Any]] = [(initial_mos, initial_ci)]
        external_starts = max(
            0, int(self.options.get("multistart_external_orbitals", 0))
        )
        active_last = template.ncore + template.ncas - 1
        for offset in range(external_starts):
            external = active_last + 1 + offset
            if external >= initial_mos.shape[1]:
                break
            swapped = initial_mos.copy()
            swapped[:, [active_last, external]] = swapped[:, [external, active_last]]
            guesses.append((swapped, None))

        candidates = []
        candidate_energies = []
        candidate_active_overlaps: list[np.ndarray | None] = []
        candidate_active_overlap_matrices: list[np.ndarray | None] = []
        candidate_root_overlaps: list[np.ndarray | None] = []
        casscf_started = time.perf_counter()
        for mo_guess, ci_guess in guesses:
            solver = configured_casscf()
            try:
                solver.kernel(mo_coeff=mo_guess, ci0=ci_guess)
                if not solver.converged:
                    solver = solver.newton().run(solver.mo_coeff, solver.ci)
            except Exception:
                continue
            if solver.converged:
                average_energy = float(np.dot(weights, np.asarray(solver.e_states)))
                candidates.append(solver)
                candidate_energies.append(average_energy)
                if (
                    previous is not None
                    and previous.get("mo_coeff") is not None
                    and previous.get("molecule") is not None
                ):
                    cross = gto.intor_cross(
                        "int1e_ovlp", previous["molecule"], molecule
                    )
                    old_active = np.asarray(previous["mo_coeff"])[
                        :, solver.ncore:solver.ncore + solver.ncas
                    ]
                    new_active = np.asarray(solver.mo_coeff)[
                        :, solver.ncore:solver.ncore + solver.ncas
                    ]
                    active_overlap = old_active.conj().T @ cross @ new_active
                    candidate_active_overlap_matrices.append(active_overlap)
                    candidate_active_overlaps.append(np.linalg.svd(
                        active_overlap, compute_uv=False,
                    ))
                    old_ci = previous.get("ci")
                    if old_ci is not None and len(old_ci) == len(solver.ci):
                        candidate_root_overlaps.append(np.asarray([
                            [
                                fci.addons.overlap(
                                    np.asarray(old), np.asarray(new),
                                    solver.ncas, solver.nelecas,
                                    s=active_overlap,
                                )
                                for new in solver.ci
                            ]
                            for old in old_ci
                        ], dtype=np.complex128))
                    else:
                        candidate_root_overlaps.append(None)
                else:
                    candidate_active_overlaps.append(None)
                    candidate_active_overlap_matrices.append(None)
                    candidate_root_overlaps.append(None)
        if not candidates:
            raise ElectronicStructureError("PySCF SA-CASSCF did not converge", retryable=True)
        timings["casscf_seconds"] = time.perf_counter() - casscf_started
        selection = str(self.options.get("orbital_selection", "energy")).lower()
        if selection == "overlap":
            if any(values is not None for values in candidate_active_overlaps):
                scores = [
                    -np.inf if values is None else float(np.min(values))
                    for values in candidate_active_overlaps
                ]
                selected_candidate = int(np.argmax(scores))
            else:
                # There is no transported subspace on the first electronic call.
                # Initialize from the lowest stationary state-averaged solution;
                # subsequent calls select by active-subspace overlap.
                selected_candidate = int(np.argmin(candidate_energies))
        elif selection == "energy":
            selected_candidate = int(np.argmin(candidate_energies))
        else:
            raise ElectronicStructureError(
                f"unsupported orbital_selection {selection!r}", retryable=False
            )
        casscf = candidates[selected_candidate]

        raw_energies = np.asarray(casscf.e_states, dtype=float)
        raw_ci = list(casscf.ci)
        spin_squares = np.asarray(
            [
                fci.spin_op.spin_square0(ci_vector, casscf.ncas, casscf.nelecas)[0]
                for ci_vector in raw_ci
            ],
            dtype=float,
        )
        # Physical state indices remain in adiabatic energy order.  Root-overlap
        # assignment is used only to diagnose when the nuclear step is too large
        # for a well-defined same-root phase.
        permutation = np.arange(len(request.states))
        assignment_suggestion = permutation.copy()
        ci_root_diagonal_overlaps = None
        phases = np.ones(len(request.states), dtype=np.complex128)
        tracking_overlap = None
        ci_coefficient_overlap = None
        active_space_singular_values = None
        active_orbital_rotation = None
        aligned_active_overlap = None
        continuity_warnings: list[dict[str, Any]] = []
        continuity_policy = str(
            self.options.get("continuity_policy", "error")
        ).lower()
        transported_mo_coeff = np.asarray(casscf.mo_coeff).copy()
        transported_ci = [np.asarray(value).copy() for value in raw_ci]
        if previous is not None and previous.get("mo_coeff") is not None:
            previous_molecule = previous.get("molecule")
            if previous_molecule is not None:
                cross_overlap = gto.intor_cross("int1e_ovlp", previous_molecule, molecule)
                old_active = np.asarray(previous["mo_coeff"])[
                    :, casscf.ncore:casscf.ncore + casscf.ncas
                ]
                new_active = np.asarray(casscf.mo_coeff)[
                    :, casscf.ncore:casscf.ncore + casscf.ncas
                ]
                active_overlap = old_active.conj().T @ cross_overlap @ new_active
                active_space_singular_values = np.linalg.svd(
                    active_overlap, compute_uv=False
                )
                if previous.get("ci") is not None and len(previous["ci"]) == len(raw_ci):
                    # CI coefficient arrays are defined in their current active-
                    # orbital basis and therefore cannot be dotted directly when
                    # active-active rotations occur.  Transform the determinants
                    # through the old/new active-orbital overlap before tracking
                    # physical, energy-ordered roots.
                    ci_coefficient_overlap = np.asarray([
                        [
                            np.vdot(
                                np.asarray(old).reshape(-1),
                                np.asarray(new).reshape(-1),
                            )
                            for new in raw_ci
                        ]
                        for old in previous["ci"]
                    ], dtype=np.complex128)
                    tracking_overlap = np.asarray([
                        [
                            fci.addons.overlap(
                                np.asarray(old), np.asarray(new),
                                casscf.ncas, casscf.nelecas,
                                s=active_overlap,
                            )
                            for new in raw_ci
                        ]
                        for old in previous["ci"]
                    ], dtype=np.complex128)
                    phases, assignment_suggestion, ci_root_diagonal_overlaps = (
                        _energy_ordered_root_phases(
                            tracking_overlap,
                            float(self.options.get("ci_root_overlap_min", 0.0)),
                            enforce=continuity_policy == "error",
                        )
                    )
                    root_minimum = float(
                        self.options.get("ci_root_overlap_min", 0.0)
                    )
                    if (
                        ci_root_diagonal_overlaps.size
                        and float(np.min(ci_root_diagonal_overlaps)) < root_minimum
                    ):
                        continuity_warnings.append({
                            "kind": "ci_root_overlap",
                            "diagonal_overlaps": ci_root_diagonal_overlaps.tolist(),
                            "assignment_suggestion": assignment_suggestion.tolist(),
                            "minimum": root_minimum,
                        })
                # Put the persisted wavefunction into the closest active-orbital
                # gauge to the preceding call.  The CI transformation is the
                # contragredient determinant representation of the same orbital
                # rotation, so this changes representation but not the physical
                # CASSCF wavefunction, energies, gradients, or NACs.
                u_active, _singular_values, vh_active = np.linalg.svd(
                    active_overlap
                )
                active_orbital_rotation = (
                    vh_active.conj().T @ u_active.conj().T
                )
                aligned_active_overlap = active_overlap @ active_orbital_rotation
                active_slice = slice(
                    casscf.ncore, casscf.ncore + casscf.ncas
                )
                transported_mo_coeff[:, active_slice] = (
                    np.asarray(casscf.mo_coeff)[:, active_slice]
                    @ active_orbital_rotation
                )
                transported_ci = [
                    np.asarray(fci.addons.transform_ci_for_orbital_rotation(
                        value, casscf.ncas, casscf.nelecas,
                        active_orbital_rotation,
                    ))
                    for value in raw_ci
                ]
                minimum_overlap = float(
                    self.options.get("active_space_overlap_min", 0.0)
                )
                if (
                    active_space_singular_values.size
                    and active_space_singular_values.min() < minimum_overlap
                ):
                    warning = {
                        "kind": "active_space_overlap",
                        "singular_values": active_space_singular_values.tolist(),
                        "minimum": minimum_overlap,
                    }
                    if continuity_policy == "error":
                        raise ElectronicStructureError(
                            "PySCF active-space continuity failure: "
                            f"singular_values={active_space_singular_values.tolist()}, "
                            f"minimum={minimum_overlap}",
                            retryable=False,
                        )
                    continuity_warnings.append(warning)

        ordered_energies = raw_energies[permutation]
        gradients = None
        gradient_mask = None
        gradient_timings: dict[str, float] = {}
        if ElectronicProperties.GRADIENTS in request.properties:
            gradients = np.full(
                (len(request.states), len(request.atoms), 3), np.nan, dtype=float
            )
            gradient_mask = np.zeros(len(request.states), dtype=bool)
            for state in request.gradient_states or request.states:
                position = request.states.index(state)
                raw_state = int(permutation[position])
                phase_started = time.perf_counter()
                gradients[position] = casscf.nuc_grad_method(state=raw_state).kernel()
                gradient_timings[str(state)] = time.perf_counter() - phase_started
                gradient_mask[position] = True
        timings["gradient_seconds"] = gradient_timings
        nacs = None
        nac_mask = None
        evaluated_nac_pairs: list[list[int]] = []
        screened_nac_pairs: list[list[int]] = []
        nac_timings: dict[str, float] = {}
        if ElectronicProperties.NACS in request.properties:
            nacs = np.zeros(
                (len(request.states), len(request.states), len(request.atoms), 3),
                dtype=np.complex128,
            )
            nac_mask = np.zeros((len(request.states), len(request.states)), dtype=bool)
            nac_method = casscf.nac_method()
            use_etfs = bool(self.options.get("use_etfs", True))
            pairs = request.nac_pairs
            if pairs is None:
                pairs = tuple(
                    (bra, ket) for offset, bra in enumerate(request.states)
                    for ket in request.states[offset + 1:]
                )
            with self._rohf_nac_context(casscf):
                for bra, ket in pairs:
                    i, j = request.states.index(bra), request.states.index(ket)
                    gap = abs(ordered_energies[i] - ordered_energies[j])
                    if request.nac_gap_threshold is not None and gap > request.nac_gap_threshold:
                        screened_nac_pairs.append([int(bra), int(ket)])
                        continue
                    raw_bra, raw_ket = int(permutation[i]), int(permutation[j])
                    phase_started = time.perf_counter()
                    # PySCF state=(ket,bra) returns <bra|d ket/dR>.
                    vector = nac_method.kernel(state=(raw_ket, raw_bra), use_etfs=use_etfs)
                    vector = phases[i].conjugate() * np.asarray(vector) * phases[j]
                    nacs[i, j] = vector
                    nacs[j, i] = -vector.conjugate()
                    nac_mask[i, j] = nac_mask[j, i] = True
                    evaluated_nac_pairs.append([int(bra), int(ket)])
                    nac_timings[f"{bra}:{ket}"] = time.perf_counter() - phase_started
        timings["nac_seconds"] = nac_timings
        energy_gradient_residuals = None
        if (
            gradients is not None
            and previous is not None
            and previous.get("energies") is not None
            and previous.get("gradients") is not None
            and previous.get("geometry") is not None
        ):
            displacement = request.geometry - np.asarray(previous["geometry"])
            actual_change = ordered_energies - np.asarray(previous["energies"])
            previous_mask = np.asarray(
                previous.get("gradient_mask", np.ones(len(request.states))), dtype=bool
            )
            available = previous_mask & np.asarray(gradient_mask, dtype=bool)
            energy_gradient_residuals = np.full(len(request.states), np.nan)
            if np.any(available):
                predicted_change = 0.5 * np.sum(
                    (np.asarray(previous["gradients"])[available] + gradients[available])
                    * displacement[None, :, :],
                    axis=(1, 2),
                )
                energy_gradient_residuals[available] = actual_change[available] - predicted_change
            tolerance = float(
                self.options.get("energy_gradient_consistency_tolerance", np.inf)
            )
            if np.any(available) and np.max(np.abs(energy_gradient_residuals[available])) > tolerance:
                warning = {
                    "kind": "energy_gradient_residual",
                    "residuals": energy_gradient_residuals.tolist(),
                    "tolerance": tolerance,
                }
                if continuity_policy == "error":
                    raise ElectronicStructureError(
                        "PySCF energy/gradient continuity failure: "
                        f"residuals={energy_gradient_residuals.tolist()}, "
                        f"tolerance={tolerance}",
                        retryable=False,
                    )
                continuity_warnings.append(warning)

        identifier = uuid.uuid4().hex
        ordered_ci = [
            np.real_if_close(np.asarray(transported_ci[index]) * phases[position])
            for position, index in enumerate(permutation)
        ]
        aligned_ci_coefficient_overlap = None
        if (
            previous is not None
            and previous.get("ci") is not None
            and len(previous["ci"]) == len(ordered_ci)
        ):
            aligned_ci_coefficient_overlap = np.asarray([
                [
                    np.vdot(
                        np.asarray(old).reshape(-1),
                        np.asarray(new).reshape(-1),
                    )
                    for new in ordered_ci
                ]
                for old in previous["ci"]
            ], dtype=np.complex128)
        payload = {
            "mo_coeff": transported_mo_coeff,
            "ci": ordered_ci,
            "molecule": molecule,
            "geometry": request.geometry.copy(),
            "energies": ordered_energies.copy(),
            "gradients": None if gradients is None else gradients.copy(),
            "gradient_mask": None if gradient_mask is None else gradient_mask.copy(),
        }
        self._states[identifier] = payload
        result = ElectronicStructureResult(
            energies=ordered_energies,
            gradients=gradients,
            nacs=nacs,
            gradient_mask=gradient_mask,
            nac_mask=nac_mask,
            wavefunction=WavefunctionState(identifier, payload),
            metadata={
                "provider": "pyscf", "pyscf_converged": True,
                "scf_initialization": scf_initialization,
                "state_permutation": permutation.tolist(),
                "root_assignment_suggestion": assignment_suggestion.tolist(),
                "ci_root_diagonal_overlaps": ci_root_diagonal_overlaps,
                "tracking_overlap": tracking_overlap,
                "ci_coefficient_overlap": ci_coefficient_overlap,
                "aligned_ci_coefficient_overlap": aligned_ci_coefficient_overlap,
                "active_space_singular_values": active_space_singular_values,
                "active_orbital_rotation": active_orbital_rotation,
                "aligned_active_overlap": aligned_active_overlap,
                "energy_gradient_residuals": energy_gradient_residuals,
                "continuity_policy": continuity_policy,
                "continuity_warnings": continuity_warnings,
                "casscf_candidate_energies": candidate_energies,
                "casscf_candidate_active_overlaps": candidate_active_overlaps,
                "casscf_candidate_active_overlap_matrices": candidate_active_overlap_matrices,
                "casscf_candidate_root_overlaps": candidate_root_overlaps,
                "casscf_selected_candidate": selected_candidate,
                "spin_squares": spin_squares[permutation].tolist(),
                "target_spin_square": target_spin_square,
                "evaluated_gradient_states": (
                    [] if gradient_mask is None else [
                        int(request.states[index]) for index in np.flatnonzero(gradient_mask)
                    ]
                ),
                "evaluated_nac_pairs": evaluated_nac_pairs,
                "screened_nac_pairs": screened_nac_pairs,
                "timings": timings,
            },
        )
        timings["total_seconds"] = time.perf_counter() - started
        return result.validate(request)

    def dump_state(self, directory: Path) -> dict[str, Any]:
        manifest: dict[str, Any] = {}
        states = self._states
        if self._checkpoint_references is not None:
            states = {
                identifier: state for identifier, state in states.items()
                if identifier in self._checkpoint_references
            }
        for identifier, state in states.items():
            filename = f"{identifier}.npz"
            ci = np.asarray(state["ci"])
            molecule_dump = state["molecule"].dumps() if state.get("molecule") is not None else ""
            np.savez_compressed(
                directory / filename,
                mo_coeff=state["mo_coeff"],
                ci=ci,
                geometry=state["geometry"],
                molecule_dump=np.asarray(molecule_dump),
                energies=state.get("energies", np.asarray([])),
                gradients=(
                    state["gradients"]
                    if state.get("gradients") is not None else np.asarray([])
                ),
                gradient_mask=(
                    state["gradient_mask"]
                    if state.get("gradient_mask") is not None else np.asarray([])
                ),
            )
            manifest[identifier] = filename
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return {"manifest": "manifest.json"}

    def load_state(self, directory: Path, metadata: dict[str, Any]) -> None:
        self._states.clear()
        manifest_name = metadata.get("manifest")
        if not manifest_name:
            return
        manifest = json.loads((directory / manifest_name).read_text())
        try:
            from pyscf import gto
        except ImportError:
            gto = None
        for identifier, filename in manifest.items():
            archive = np.load(directory / filename)
            molecule_dump = str(archive["molecule_dump"]) if "molecule_dump" in archive else ""
            self._states[identifier] = {
                "mo_coeff": archive["mo_coeff"],
                "ci": [value for value in archive["ci"]],
                "geometry": archive["geometry"],
                "molecule": gto.loads(molecule_dump) if gto is not None and molecule_dump else None,
                "energies": archive["energies"] if "energies" in archive and archive["energies"].size else None,
                "gradients": archive["gradients"] if "gradients" in archive and archive["gradients"].size else None,
                "gradient_mask": (
                    archive["gradient_mask"]
                    if "gradient_mask" in archive and archive["gradient_mask"].size else None
                ),
            }


def from_config(config) -> PySCFProvider:
    return PySCFProvider(
        basis=config.basis,
        charge=config.charge,
        spin=config.spin,
        scf_method=config.scf_method,
        active_electrons=config.active_electrons,
        active_orbitals=config.active_orbitals,
        state_weights=config.state_weights,
        options=config.provider_option_dict(),
    )
