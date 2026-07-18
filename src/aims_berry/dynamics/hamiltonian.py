"""Hamiltonian construction strategies for Gaussian trajectory bases."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.linalg import logm, polar
from scipy.integrate import quad

from ..core import MatrixSet, TrajectoryBasisFunction
from ..electronic.base import ElectronicStructureResult
from .gaussian import (
    gaussian_kinetic,
    gaussian_momentum,
    gaussian_overlap,
    gaussian_sdot,
    momentum_1d,
    overlap_1d,
    overlap_matrix,
    pair_centroid,
)


def npi_time_derivative(overlap: np.ndarray, dt: float) -> np.ndarray:
    unitary, _ = polar(np.asarray(overlap, dtype=np.complex128))
    derivative = logm(unitary) / dt
    return 0.5 * (derivative - derivative.conj().T)


class SaddlePointHamiltonian:
    """Zeroth-order saddle-point AIMS Hamiltonian."""

    def __init__(self, coupling_mode: str = "nac", pair_overlap_threshold: float = 0.0) -> None:
        self.coupling_mode = coupling_mode
        self.pair_overlap_threshold = float(pair_overlap_threshold)

    def build(
        self,
        trajectories: list[TrajectoryBasisFunction],
        centroid_evaluator: Callable[
            [TrajectoryBasisFunction, TrajectoryBasisFunction, np.ndarray],
            ElectronicStructureResult,
        ],
        velocities: list[np.ndarray],
        forces: list[np.ndarray],
        dt: float,
    ) -> MatrixSet:
        count = len(trajectories)
        overlap = overlap_matrix(trajectories)
        hamiltonian = np.zeros((count, count), dtype=np.complex128)
        sdot = np.zeros((count, count), dtype=np.complex128)
        for i, left in enumerate(trajectories):
            for j, right in enumerate(trajectories):
                sdot[i, j] = gaussian_sdot(
                    left, right, velocities[j], forces[j]
                )
        for i, left in enumerate(trajectories):
            for j in range(i, count):
                right = trajectories[j]
                nuclear_overlap = gaussian_overlap(left, right, electronic=False)
                if i != j and abs(nuclear_overlap) < self.pair_overlap_threshold:
                    overlap[i, j] = overlap[j, i] = 0.0
                    sdot[i, j] = sdot[j, i] = 0.0
                    continue
                centroid = pair_centroid(left, right)
                if i == j and right.electronic is not None:
                    electronic = right.electronic
                else:
                    electronic = centroid_evaluator(left, right, centroid)
                state_i, state_j = left.state, right.state
                if state_i == state_j:
                    value = gaussian_kinetic(left, right) + electronic.energies[state_i] * nuclear_overlap
                else:
                    value = 0j
                if (
                    self.coupling_mode == "nac"
                    and electronic.nacs is not None
                    and (
                        electronic.nac_mask is None
                        or electronic.nac_mask[state_i, state_j]
                    )
                ):
                    momentum = gaussian_momentum(left, right)
                    derivative = electronic.nacs[state_i, state_j]
                    # PySpawn Eq. (8)/(11)/(16): H_IJ contains 2D_IJ with
                    # D_IJ=(1/2M)d_IJ.<d/dR>. Since <d/dR>=i<p> for
                    # p=-i d/dR, the complete first-derivative term is +i d.p/M.
                    value += 1j * np.sum(derivative * momentum / right.masses)
                elif self.coupling_mode == "npi" and electronic.state_overlaps is not None:
                    tdc = npi_time_derivative(electronic.state_overlaps, dt)
                    sdot[i, j] += tdc[state_i, state_j] * nuclear_overlap
                    if i != j:
                        # Nuclear tau=<chi_i|dot chi_j> is one-sided, but the
                        # electronic state-transport generator is
                        # anti-Hermitian.  Populate its reverse ordered matrix
                        # element explicitly; omitting it makes H-i*tau
                        # one-way and suppresses the reciprocal population
                        # transfer present in PySpawn's NPI Hamiltonian.
                        sdot[j, i] += (
                            tdc[state_j, state_i] * nuclear_overlap.conjugate()
                        )
                hamiltonian[i, j] = value
                hamiltonian[j, i] = value.conjugate()
        hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
        return MatrixSet(overlap=overlap, hamiltonian=hamiltonian, sdot=sdot)


def _complex_quad(function, lower=-np.inf, upper=np.inf, tolerance=1.0e-10) -> complex:
    real = quad(lambda value: float(np.real(function(value))), lower, upper, epsabs=tolerance, limit=200)[0]
    imag = quad(lambda value: float(np.imag(function(value))), lower, upper, epsabs=tolerance, limit=200)[0]
    return complex(real, imag)


class BerryExactHamiltonian:
    """Semianalytic prototype Hamiltonian for the two-dimensional Berry model.

    Frozen-Gaussian kinetic, momentum, and spectator-coordinate factors are analytic;
    the only numerical integral is the localized x-dependent electronic factor.
    """

    def __init__(self, model, tolerance: float = 1.0e-10) -> None:
        self.model = model
        self.tolerance = tolerance

    def _x_weighted(self, left, right, function) -> complex:
        r1, p1, a1 = left.positions.flat[0], left.momenta.flat[0], left.widths.flat[0]
        r2, p2, a2 = right.positions.flat[0], right.momenta.flat[0], right.widths.flat[0]
        n1 = (2 * a1 / np.pi) ** 0.25
        n2 = (2 * a2 / np.pi) ** 0.25

        def integrand(x):
            product = n1 * n2 * np.exp(
                -a1 * (x - r1) ** 2 - a2 * (x - r2) ** 2
                + 1j * (p2 * (x - r2) - p1 * (x - r1))
                + 1j * (right.phase - left.phase) / left.ndof
            )
            return product * function(float(x))

        return _complex_quad(integrand, tolerance=self.tolerance)

    def _electronic_element(self, left, right) -> complex:
        one_d = [
            overlap_1d(r1, p1, a1, left.phase / left.ndof, r2, p2, a2, right.phase / right.ndof)
            for r1, p1, a1, r2, p2, a2 in zip(
                left.positions.flat, left.momenta.flat, left.widths.flat,
                right.positions.flat, right.momenta.flat, right.widths.flat,
            )
        ]
        spectator = complex(np.prod(one_d[1:]))
        x_overlap = one_d[0]
        bra, ket = left.state, right.state
        value = gaussian_kinetic(left, right) if bra == ket else 0j
        if bra == ket:
            value += (-self.model.energy if bra == 0 else self.model.energy) * x_overlap * spectator

        masses = right.masses
        template = right.positions.copy()

        def scalar_at(x):
            point = template.copy()
            point.reshape(-1)[0] = x
            return self.model.geometric_scalar(point, masses, bra, ket)

        value += self._x_weighted(left, right, scalar_at) * spectator

        # -i d.p/m, with the x-dependent factors integrated and y/z analytic.
        def dx_momentum_at(x):
            point = template.copy()
            point.reshape(-1)[0] = x
            derivative = self.model.derivative_coupling(point, bra, ket).reshape(-1)[0]
            local_p = right.momenta.flat[0] + 2j * right.widths.flat[0] * (x - right.positions.flat[0])
            return derivative * local_p / right.masses.flat[0]

        value += -1j * self._x_weighted(left, right, dx_momentum_at) * spectator
        for axis in (1, 2):
            other = complex(np.prod([one_d[k] for k in range(1, len(one_d)) if k != axis]))
            p_axis = momentum_1d(
                left.positions.flat[axis], left.momenta.flat[axis], left.widths.flat[axis],
                right.positions.flat[axis], right.momenta.flat[axis], right.widths.flat[axis],
                one_d[axis],
            )

            def derivative_at(x, component=axis):
                point = template.copy()
                point.reshape(-1)[0] = x
                return self.model.derivative_coupling(point, bra, ket).reshape(-1)[component]

            value += -1j * self._x_weighted(left, right, derivative_at) * p_axis * other / right.masses.flat[axis]
        return value

    def build(self, trajectories, _centroid_evaluator, velocities, forces, _dt) -> MatrixSet:
        count = len(trajectories)
        overlap = overlap_matrix(trajectories)
        hamiltonian = np.zeros((count, count), dtype=np.complex128)
        sdot = np.zeros((count, count), dtype=np.complex128)
        for i, left in enumerate(trajectories):
            for j, right in enumerate(trajectories):
                sdot[i, j] = gaussian_sdot(left, right, velocities[j], forces[j])
            for j in range(i, count):
                value = self._electronic_element(left, trajectories[j])
                hamiltonian[i, j] = value
                hamiltonian[j, i] = value.conjugate()
        return MatrixSet(overlap, 0.5 * (hamiltonian + hamiltonian.conj().T), sdot)
