"""PySCF SA-CASSCF electronic-structure provider."""

from __future__ import annotations

import json
import uuid
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
        mean_field.kernel()
        if not mean_field.converged:
            raise ElectronicStructureError("PySCF SCF did not converge", retryable=True)
        casscf = mcscf.CASSCF(mean_field, self.active_orbitals, self.active_electrons)
        casscf.conv_tol = float(self.options.get("casscf_conv_tol", 1.0e-8))
        casscf.max_cycle_macro = int(self.options.get("casscf_max_cycle", 50))
        target_spin = 0.5 * molecule.spin
        target_spin_square = float(
            self.options.get("spin_square", target_spin * (target_spin + 1.0))
        )
        if bool(self.options.get("fix_spin", True)):
            casscf.fix_spin_(
                shift=float(self.options.get("spin_penalty", 0.5)),
                ss=target_spin_square,
            )
        weights = self.state_weights or tuple([1.0 / len(request.states)] * len(request.states))
        casscf = casscf.state_average(weights)
        previous = self._previous_payload(request.previous)
        kwargs: dict[str, Any] = {}
        if previous is not None and previous.get("mo_coeff") is not None:
            try:
                kwargs["mo_coeff"] = mcscf.addons.project_init_guess(
                    casscf, previous["mo_coeff"], prev_mol=previous.get("molecule")
                )
                kwargs["ci0"] = previous.get("ci")
            except Exception:
                kwargs = {}
        casscf.kernel(**kwargs)
        if not casscf.converged:
            # A second-order fallback is intentionally finite and explicit.
            try:
                casscf = casscf.newton().run(casscf.mo_coeff, casscf.ci)
            except Exception as exc:
                raise ElectronicStructureError("PySCF SA-CASSCF did not converge", retryable=True) from exc
        if not casscf.converged:
            raise ElectronicStructureError("PySCF SA-CASSCF did not converge", retryable=True)

        raw_energies = np.asarray(casscf.e_states, dtype=float)
        raw_ci = list(casscf.ci)
        spin_squares = np.asarray(
            [
                fci.spin_op.spin_square0(ci_vector, casscf.ncas, casscf.nelecas)[0]
                for ci_vector in raw_ci
            ],
            dtype=float,
        )
        permutation = np.arange(len(request.states))
        phases = np.ones(len(request.states), dtype=np.complex128)
        tracking_overlap = None
        if previous is not None and previous.get("ci") is not None and len(previous["ci"]) == len(raw_ci):
            tracking_overlap = np.asarray([
                [np.vdot(np.asarray(old).reshape(-1), np.asarray(new).reshape(-1)) for new in raw_ci]
                for old in previous["ci"]
            ], dtype=np.complex128)
            row, col = linear_sum_assignment(-np.abs(tracking_overlap))
            permutation = col[np.argsort(row)]
            diagonal = tracking_overlap[np.arange(len(permutation)), permutation]
            phases = np.exp(-1j * np.angle(np.where(abs(diagonal) > 1.0e-14, diagonal, 1.0)))

        gradients = None
        if ElectronicProperties.GRADIENTS in request.properties:
            gradients_raw = np.asarray([
                casscf.nuc_grad_method(state=index).kernel()
                for index in range(len(request.states))
            ], dtype=float)
            gradients = gradients_raw[permutation]
        nacs = None
        if ElectronicProperties.NACS in request.properties:
            raw_nacs = np.zeros((len(request.states), len(request.states), len(request.atoms), 3), dtype=np.complex128)
            nac_method = casscf.nac_method()
            use_etfs = bool(self.options.get("use_etfs", True))
            with self._rohf_nac_context(casscf):
                for bra in range(len(request.states)):
                    for ket in range(bra + 1, len(request.states)):
                        # PySCF state=(ket,bra) returns <bra|d ket/dR>.
                        vector = nac_method.kernel(state=(ket, bra), use_etfs=use_etfs)
                        raw_nacs[bra, ket] = vector
                        raw_nacs[ket, bra] = -np.asarray(vector).conjugate()
            nacs = raw_nacs[permutation][:, permutation]
            nacs = phases.conj()[:, None, None, None] * nacs * phases[None, :, None, None]

        identifier = uuid.uuid4().hex
        ordered_ci = [
            np.real_if_close(np.asarray(raw_ci[index]) * phases[position])
            for position, index in enumerate(permutation)
        ]
        payload = {
            "mo_coeff": np.asarray(casscf.mo_coeff),
            "ci": ordered_ci,
            "molecule": molecule,
            "geometry": request.geometry.copy(),
        }
        self._states[identifier] = payload
        result = ElectronicStructureResult(
            energies=raw_energies[permutation],
            gradients=gradients,
            nacs=nacs,
            wavefunction=WavefunctionState(identifier, payload),
            metadata={
                "provider": "pyscf", "pyscf_converged": True,
                "state_permutation": permutation.tolist(),
                "tracking_overlap": tracking_overlap,
                "spin_squares": spin_squares[permutation].tolist(),
                "target_spin_square": target_spin_square,
            },
        )
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
