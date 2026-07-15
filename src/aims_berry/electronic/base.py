"""Backend-neutral electronic-structure interfaces."""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


class ElectronicProperties(enum.StrEnum):
    ENERGIES = "energies"
    GRADIENTS = "gradients"
    NACS = "nacs"
    STATE_OVERLAPS = "state_overlaps"
    DIPOLES = "dipoles"
    CHARGES = "charges"


@dataclasses.dataclass(frozen=True)
class ProviderCapabilities:
    energies: bool = True
    gradients: bool = True
    nacs: bool = False
    state_overlaps: bool = False
    npi_tdc: bool = False
    complex_values: bool = False
    berry_connection: bool = False
    berry_curvature: bool = False
    hessian: bool = False
    dipoles: bool = False
    charges: bool = False

    def require(self, properties: frozenset[ElectronicProperties]) -> None:
        missing = [p.value for p in properties if not getattr(self, p.value)]
        if missing:
            raise ProviderCapabilityError(f"provider lacks: {', '.join(sorted(missing))}")


@dataclasses.dataclass(frozen=True)
class WavefunctionState:
    """Serializable or provider-owned state used as the next electronic guess."""

    identifier: str
    payload: Any = None
    artifacts: tuple[Path, ...] = ()


@dataclasses.dataclass(frozen=True)
class ElectronicStructureRequest:
    atoms: tuple[str, ...]
    atomic_numbers: np.ndarray
    geometry: np.ndarray
    states: tuple[int, ...]
    properties: frozenset[ElectronicProperties]
    active_state: int
    time: float
    previous: WavefunctionState | None = None
    gradient_states: tuple[int, ...] | None = None
    nac_pairs: tuple[tuple[int, int], ...] | None = None
    nac_gap_threshold: float | None = None

    def __post_init__(self) -> None:
        xyz = np.asarray(self.geometry, dtype=float)
        numbers = np.asarray(self.atomic_numbers, dtype=int)
        if xyz.shape != (len(self.atoms), 3):
            raise ValueError("geometry must have shape (natom, 3)")
        if numbers.shape != (len(self.atoms),):
            raise ValueError("atomic_numbers must have shape (natom,)")
        if self.active_state not in self.states:
            raise ValueError("active_state must be requested")
        requested = set(self.states)
        if self.gradient_states is not None and not set(self.gradient_states) <= requested:
            raise ValueError("gradient_states must be a subset of states")
        if self.nac_pairs is not None:
            for bra, ket in self.nac_pairs:
                if bra not in requested or ket not in requested or bra == ket:
                    raise ValueError("nac_pairs must contain distinct requested states")
        if self.nac_gap_threshold is not None and self.nac_gap_threshold <= 0:
            raise ValueError("nac_gap_threshold must be positive")


@dataclasses.dataclass
class ElectronicStructureResult:
    energies: np.ndarray
    gradients: np.ndarray | None = None
    nacs: np.ndarray | None = None
    state_overlaps: np.ndarray | None = None
    dipoles: np.ndarray | None = None
    charges: np.ndarray | None = None
    wavefunction: WavefunctionState | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    gradient_mask: np.ndarray | None = None
    nac_mask: np.ndarray | None = None

    def gradient_at_state(self, state_index: int) -> np.ndarray:
        if self.gradients is None or self.gradient_mask is None or not self.gradient_mask[state_index]:
            raise ElectronicStructureError(f"gradient for state index {state_index} is unavailable")
        return self.gradients[state_index]

    def nac_between(self, bra_index: int, ket_index: int) -> np.ndarray:
        if self.nacs is None or self.nac_mask is None or not self.nac_mask[bra_index, ket_index]:
            raise ElectronicStructureError(
                f"NAC for state indices ({bra_index}, {ket_index}) is unavailable"
            )
        return self.nacs[bra_index, ket_index]

    def validate(self, request: ElectronicStructureRequest, atol: float = 1.0e-8) -> "ElectronicStructureResult":
        nstate, natom = len(request.states), len(request.atoms)
        self.energies = np.asarray(self.energies, dtype=float)
        if self.energies.shape != (nstate,):
            raise ElectronicStructureError(f"energies must have shape {(nstate,)}")
        if not np.all(np.isfinite(self.energies)):
            raise ElectronicStructureError("energies contain non-finite values")
        if self.gradients is not None:
            self.gradients = np.asarray(self.gradients, dtype=float)
            if self.gradients.shape != (nstate, natom, 3):
                raise ElectronicStructureError(f"gradients must have shape {(nstate, natom, 3)}")
            if self.gradient_mask is None:
                self.gradient_mask = np.ones(nstate, dtype=bool)
            self.gradient_mask = np.asarray(self.gradient_mask, dtype=bool)
            if self.gradient_mask.shape != (nstate,):
                raise ElectronicStructureError(f"gradient_mask must have shape {(nstate,)}")
            if not np.all(np.isfinite(self.gradients[self.gradient_mask])):
                raise ElectronicStructureError("requested gradients contain non-finite values")
        if self.nacs is not None:
            self.nacs = np.asarray(self.nacs, dtype=np.complex128)
            if self.nacs.shape != (nstate, nstate, natom, 3):
                raise ElectronicStructureError(f"nacs must have shape {(nstate, nstate, natom, 3)}")
            if not np.all(np.isfinite(self.nacs)):
                raise ElectronicStructureError("NACs contain non-finite values")
            if self.nac_mask is None:
                self.nac_mask = np.ones((nstate, nstate), dtype=bool)
                np.fill_diagonal(self.nac_mask, False)
            self.nac_mask = np.asarray(self.nac_mask, dtype=bool)
            if self.nac_mask.shape != (nstate, nstate):
                raise ElectronicStructureError(f"nac_mask must have shape {(nstate, nstate)}")
            if not np.array_equal(self.nac_mask, self.nac_mask.T):
                raise ElectronicStructureError("nac_mask must be symmetric")
            residual = self.nacs + self.nacs.swapaxes(0, 1).conj()
            if np.linalg.norm(residual) > atol * max(1.0, np.linalg.norm(self.nacs)):
                raise ElectronicStructureError("NAC tensor is not anti-Hermitian in its state indices")
        if self.state_overlaps is not None:
            self.state_overlaps = np.asarray(self.state_overlaps, dtype=np.complex128)
            if self.state_overlaps.shape != (nstate, nstate):
                raise ElectronicStructureError(f"state_overlaps must have shape {(nstate, nstate)}")
        requested = request.properties
        if ElectronicProperties.GRADIENTS in requested:
            if self.gradients is None:
                raise ElectronicStructureError("provider omitted requested gradients")
            required = request.gradient_states or request.states
            indices = [request.states.index(state) for state in required]
            if not np.all(self.gradient_mask[indices]):
                raise ElectronicStructureError("provider omitted a requested state gradient")
        if ElectronicProperties.NACS in requested:
            if self.nacs is None:
                raise ElectronicStructureError("provider omitted requested NACs")
            pairs = request.nac_pairs
            if pairs is None:
                pairs = tuple(
                    (bra, ket) for offset, bra in enumerate(request.states)
                    for ket in request.states[offset + 1:]
                )
            for bra, ket in pairs:
                i, j = request.states.index(bra), request.states.index(ket)
                gap = abs(self.energies[i] - self.energies[j])
                screened = request.nac_gap_threshold is not None and gap > request.nac_gap_threshold
                if not screened and not self.nac_mask[i, j]:
                    raise ElectronicStructureError("provider omitted a requested NAC pair")
        if ElectronicProperties.STATE_OVERLAPS in requested and self.state_overlaps is None:
            raise ElectronicStructureError("provider omitted requested state overlaps")
        return self


class ElectronicStructureError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class ProviderCapabilityError(ElectronicStructureError):
    pass


@runtime_checkable
class ElectronicStructureProvider(Protocol):
    @property
    def capabilities(self) -> ProviderCapabilities: ...

    def evaluate(self, request: ElectronicStructureRequest) -> ElectronicStructureResult: ...

    def dump_state(self, directory: Path) -> dict[str, Any]: ...

    def load_state(self, directory: Path, metadata: dict[str, Any]) -> None: ...


class BaseProvider:
    capabilities = ProviderCapabilities()

    def dump_state(self, directory: Path) -> dict[str, Any]:
        return {}

    def load_state(self, directory: Path, metadata: dict[str, Any]) -> None:
        return None
