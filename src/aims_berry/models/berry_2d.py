"""Complex two-state Berry model migrated from the research prototype."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erf

from ..electronic.base import (
    BaseProvider,
    ElectronicProperties,
    ElectronicStructureRequest,
    ElectronicStructureResult,
    ProviderCapabilities,
)


class BerryModel2DParallelTransport(BaseProvider):
    """Two-state adiabatic model with a nontrivial complex Berry connection."""

    capabilities = ProviderCapabilities(
        energies=True,
        gradients=True,
        nacs=True,
        state_overlaps=True,
        npi_tdc=True,
        complex_values=True,
        berry_connection=True,
        berry_curvature=True,
    )

    def __init__(self, energy: float = 0.02, sharpness: float = 3.0, phase_gradient: float = 5.0):
        self.energy = float(energy)
        self.sharpness = float(sharpness)
        self.phase_gradient = float(phase_gradient)
        self._last_vectors: np.ndarray | None = None

    def theta(self, x: float) -> float:
        return 0.5 * math.pi * (float(erf(self.sharpness * x)) + 1.0)

    def theta_prime(self, x: float) -> float:
        return math.sqrt(math.pi) * self.sharpness * math.exp(-(self.sharpness * x) ** 2)

    def theta_second(self, x: float) -> float:
        return -2 * math.sqrt(math.pi) * self.sharpness**3 * x * math.exp(-(self.sharpness * x) ** 2)

    def adiabatic_vectors(self, point: np.ndarray) -> np.ndarray:
        x, y = np.asarray(point).reshape(-1)[:2]
        theta = self.theta(float(x))
        phase = self.phase_gradient * float(y)
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        return np.column_stack((
            np.array([c * np.exp(1j * phase), -s]),
            np.array([s * np.exp(1j * phase), c]),
        )).astype(np.complex128)

    def nac_vector(self, point: np.ndarray, bra: int, ket: int) -> np.ndarray:
        x = float(np.asarray(point).reshape(-1)[0])
        theta = self.theta(x)
        value = np.array([
            0.5 * self.theta_prime(x),
            0.5j * math.sin(theta) * self.phase_gradient,
            0.0,
        ], dtype=np.complex128)
        return value if bra < ket else -value.conjugate()

    def derivative_coupling(self, positions: np.ndarray, bra: int, ket: int) -> np.ndarray:
        """Return ``<bra|nabla|ket>`` in the prototype's parallel-transport gauge."""
        if bra == ket:
            return 1j * self.connection(positions, bra)
        result = np.zeros_like(np.asarray(positions), dtype=np.complex128)
        result.reshape(-1)[:3] = self.nac_vector(positions, bra, ket)
        return result

    def divergence_coupling(
        self, positions: np.ndarray, masses: np.ndarray, bra: int, ket: int
    ) -> complex:
        """Mass-weighted divergence of the derivative coupling."""
        x = float(np.asarray(positions).reshape(-1)[0])
        if bra == ket:
            return 0j
        sign = 1.0 if bra < ket else -1.0
        return complex(sign * 0.5 * self.theta_second(x) / np.asarray(masses).reshape(-1)[0])

    def geometric_scalar(
        self, positions: np.ndarray, masses: np.ndarray, bra: int, ket: int
    ) -> complex:
        """Born-Huang/divergence term from the exact adiabatic kinetic operator."""
        contraction = 0j
        inverse_mass = 1.0 / np.asarray(masses)
        for intermediate in range(2):
            left = self.derivative_coupling(positions, bra, intermediate)
            right = self.derivative_coupling(positions, intermediate, ket)
            contraction += np.sum(left * inverse_mass * right)
        return -0.5 * (self.divergence_coupling(positions, masses, bra, ket) + contraction)

    def connection(self, positions: np.ndarray, state: int) -> np.ndarray:
        x = float(np.asarray(positions).reshape(-1)[0])
        theta = self.theta(x)
        ay = self.phase_gradient * (math.cos(theta / 2) ** 2 if state == 0 else math.sin(theta / 2) ** 2)
        result = np.zeros_like(np.asarray(positions), dtype=float)
        result.reshape(-1)[1] = ay
        return result

    def curvature(self, positions: np.ndarray, state: int) -> np.ndarray:
        x = float(np.asarray(positions).reshape(-1)[0])
        theta = self.theta(x)
        sign = -1.0 if state == 0 else 1.0
        day_dx = sign * 0.5 * math.sin(theta) * self.theta_prime(x) * self.phase_gradient
        ndof = np.asarray(positions).size
        omega = np.zeros((ndof, ndof), dtype=float)
        omega[0, 1] = day_dx
        omega[1, 0] = -day_dx
        return omega

    def scalar_gradient(self, positions: np.ndarray, masses: np.ndarray, state: int) -> np.ndarray:
        point = np.asarray(positions).reshape(-1)
        mass = np.asarray(masses).reshape(-1)
        x = float(point[0])
        theta = self.theta(x)
        tp = self.theta_prime(x)
        tpp = self.theta_second(x)
        mx, my = mass[:2]
        common = tp * tpp / (4 * mx) + self.phase_gradient**2 * math.sin(theta) * math.cos(theta) * tp / (4 * my)
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        state_term = (-self.phase_gradient**2 * c**3 * s * tp / my if state == 0 else self.phase_gradient**2 * s**3 * c * tp / my)
        full_d = common + state_term
        connection = self.connection(positions, state).reshape(-1)
        jacobian_term = (self.curvature(positions, state)[0, 1] * connection[1] / my)
        result = np.zeros_like(point, dtype=float)
        result[0] = full_d - jacobian_term
        return result.reshape(np.asarray(positions).shape)

    def classical_energy(self, positions: np.ndarray, canonical_momenta: np.ndarray, masses: np.ndarray, state: int) -> float:
        connection = self.connection(positions, state)
        kinetic = 0.5 * np.sum((canonical_momenta + connection) ** 2 / masses)
        return float(kinetic + (-self.energy if state == 0 else self.energy))

    def evaluate(self, request: ElectronicStructureRequest) -> ElectronicStructureResult:
        self.capabilities.require(request.properties)
        point = request.geometry.reshape(-1)
        states = request.states
        energies = np.asarray([-self.energy if s == 0 else self.energy for s in states], dtype=float)
        gradients = np.zeros((len(states), len(request.atoms), 3), dtype=float)
        nacs = np.zeros((len(states), len(states), len(request.atoms), 3), dtype=np.complex128)
        for i, bra in enumerate(states):
            for j, ket in enumerate(states):
                if bra != ket:
                    nacs[i, j, 0] = self.nac_vector(point, bra, ket)
                else:
                    nacs[i, i, 0] = 1j * self.connection(request.geometry, bra)[0]
        vectors = self.adiabatic_vectors(point)
        overlaps = None if self._last_vectors is None else self._last_vectors.conj().T @ vectors
        self._last_vectors = vectors
        if overlaps is None and ElectronicProperties.STATE_OVERLAPS in request.properties:
            overlaps = np.eye(len(states), dtype=np.complex128)
        return ElectronicStructureResult(
            energies=energies,
            gradients=gradients,
            nacs=nacs,
            state_overlaps=overlaps,
            metadata={"provider": "berry_2d"},
        ).validate(request)

    def dump_state(self, directory: Path) -> dict[str, Any]:
        if self._last_vectors is not None:
            np.save(directory / "berry_vectors.npy", self._last_vectors)
            return {"vectors": "berry_vectors.npy"}
        return {}

    def load_state(self, directory: Path, metadata: dict[str, Any]) -> None:
        if "vectors" in metadata:
            self._last_vectors = np.load(directory / metadata["vectors"])
