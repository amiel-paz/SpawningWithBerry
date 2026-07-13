"""Analytic frozen-Gaussian matrix elements."""

from __future__ import annotations

import numpy as np

from ..core import TrajectoryBasisFunction


def overlap_1d(r1: float, p1: float, a1: float, g1: float, r2: float, p2: float, a2: float, g2: float) -> complex:
    total_a = a1 + a2
    linear = 2 * a1 * r1 + 2 * a2 * r2 + 1j * (p2 - p1)
    constant = -a1 * r1**2 - a2 * r2**2 - 1j * p2 * r2 + 1j * p1 * r1 + 1j * (g2 - g1)
    normalization = (2 * a1 / np.pi) ** 0.25 * (2 * a2 / np.pi) ** 0.25
    return complex(normalization * np.sqrt(np.pi / total_a) * np.exp(constant + linear**2 / (4 * total_a)))


def _product_moments(r1: float, p1: float, a1: float, r2: float, p2: float, a2: float) -> tuple[complex, complex]:
    total_a = a1 + a2
    linear = 2 * a1 * r1 + 2 * a2 * r2 + 1j * (p2 - p1)
    mean_x = linear / (2 * total_a)
    variance = 1 / (2 * total_a)
    return mean_x, variance


def momentum_1d(r1: float, p1: float, a1: float, r2: float, p2: float, a2: float, overlap: complex) -> complex:
    mean_x, _ = _product_moments(r1, p1, a1, r2, p2, a2)
    return overlap * (p2 + 2j * a2 * (mean_x - r2))


def kinetic_1d(r1: float, p1: float, a1: float, r2: float, p2: float, a2: float, mass: float, overlap: complex) -> complex:
    mean_x, variance = _product_moments(r1, p1, a1, r2, p2, a2)
    delta = mean_x - r2
    p2_expect = 2 * a2 + p2**2 + 4j * a2 * p2 * delta - 4 * a2**2 * (delta**2 + variance)
    return overlap * p2_expect / (2 * mass)


def gaussian_overlap(left: TrajectoryBasisFunction, right: TrajectoryBasisFunction, *, electronic: bool = True) -> complex:
    if electronic and left.state != right.state:
        return 0j
    values = [
        overlap_1d(r1, p1, a1, left.phase / left.ndof, r2, p2, a2, right.phase / right.ndof)
        for r1, p1, a1, r2, p2, a2 in zip(
            left.positions.flat, left.momenta.flat, left.widths.flat,
            right.positions.flat, right.momenta.flat, right.widths.flat,
        )
    ]
    return complex(np.prod(values))


def gaussian_momentum(left: TrajectoryBasisFunction, right: TrajectoryBasisFunction) -> np.ndarray:
    one_d = [
        overlap_1d(r1, p1, a1, left.phase / left.ndof, r2, p2, a2, right.phase / right.ndof)
        for r1, p1, a1, r2, p2, a2 in zip(
            left.positions.flat, left.momenta.flat, left.widths.flat,
            right.positions.flat, right.momenta.flat, right.widths.flat,
        )
    ]
    output = np.empty(left.ndof, dtype=np.complex128)
    for k, (r1, p1, a1, r2, p2, a2) in enumerate(zip(
        left.positions.flat, left.momenta.flat, left.widths.flat,
        right.positions.flat, right.momenta.flat, right.widths.flat,
    )):
        output[k] = momentum_1d(r1, p1, a1, r2, p2, a2, one_d[k]) * np.prod(one_d[:k] + one_d[k + 1:])
    return output.reshape(left.positions.shape)


def gaussian_kinetic(left: TrajectoryBasisFunction, right: TrajectoryBasisFunction) -> complex:
    if left.state != right.state:
        return 0j
    one_d = [
        overlap_1d(r1, p1, a1, left.phase / left.ndof, r2, p2, a2, right.phase / right.ndof)
        for r1, p1, a1, r2, p2, a2 in zip(
            left.positions.flat, left.momenta.flat, left.widths.flat,
            right.positions.flat, right.momenta.flat, right.widths.flat,
        )
    ]
    total = 0j
    for k, (r1, p1, a1, r2, p2, a2, mass) in enumerate(zip(
        left.positions.flat, left.momenta.flat, left.widths.flat,
        right.positions.flat, right.momenta.flat, right.widths.flat, right.masses.flat,
    )):
        total += kinetic_1d(r1, p1, a1, r2, p2, a2, mass, one_d[k]) * np.prod(one_d[:k] + one_d[k + 1:])
    return complex(total)


def gaussian_sdot(
    left: TrajectoryBasisFunction,
    right: TrajectoryBasisFunction,
    rdot: np.ndarray,
    pdot: np.ndarray,
    gamma_dot: float = 0.0,
) -> complex:
    if left.state != right.state:
        return 0j
    one_d = []
    derivatives = []
    for r1, p1, a1, r2, p2, a2, vr, vp in zip(
        left.positions.flat, left.momenta.flat, left.widths.flat,
        right.positions.flat, right.momenta.flat, right.widths.flat,
        np.asarray(rdot).flat, np.asarray(pdot).flat,
    ):
        s = overlap_1d(r1, p1, a1, left.phase / left.ndof, r2, p2, a2, right.phase / right.ndof)
        mean_x, _ = _product_moments(r1, p1, a1, r2, p2, a2)
        derivatives.append(s * (2 * a2 * (mean_x - r2) * vr + 1j * (vp * (mean_x - r2) - p2 * vr + gamma_dot / right.ndof)))
        one_d.append(s)
    return complex(sum(d * np.prod(one_d[:k] + one_d[k + 1:]) for k, d in enumerate(derivatives)))


def overlap_matrix(trajectories: list[TrajectoryBasisFunction]) -> np.ndarray:
    n = len(trajectories)
    matrix = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            matrix[i, j] = gaussian_overlap(trajectories[i], trajectories[j])
            matrix[j, i] = matrix[i, j].conjugate()
    return matrix


def pair_centroid(left: TrajectoryBasisFunction, right: TrajectoryBasisFunction) -> np.ndarray:
    weights = left.widths + right.widths
    return (left.widths * left.positions + right.widths * right.positions) / weights
