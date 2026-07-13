"""Classical centroid propagation, including Berry-curvature forces."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def velocity_verlet(
    positions: np.ndarray,
    momenta: np.ndarray,
    masses: np.ndarray,
    dt: float,
    gradient: np.ndarray,
    gradient_at: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    force0 = -np.asarray(gradient, dtype=float)
    half = momenta + 0.5 * dt * force0
    new_positions = positions + dt * half / masses
    gradient1 = np.asarray(gradient_at(new_positions), dtype=float)
    new_momenta = half - 0.5 * dt * gradient1
    return new_positions, new_momenta, gradient1


def _boris_rotate(velocity: np.ndarray, curvature: np.ndarray, masses: np.ndarray, dt: float) -> np.ndarray:
    flat_v = np.asarray(velocity, dtype=float).reshape(-1)
    flat_m = np.asarray(masses, dtype=float).reshape(-1)
    omega = np.asarray(curvature, dtype=float).reshape(flat_v.size, flat_v.size)
    scaled = omega / np.sqrt(flat_m[:, None] * flat_m[None, :])
    transform = -0.5 * dt * scaled
    identity = np.eye(flat_v.size)
    mass_v = np.sqrt(flat_m) * flat_v
    rotated = np.linalg.solve(identity - transform, (identity + transform) @ mass_v)
    return (rotated / np.sqrt(flat_m)).reshape(velocity.shape)


def berry_boris_step(
    positions: np.ndarray,
    canonical_momenta: np.ndarray,
    masses: np.ndarray,
    state: int,
    dt: float,
    connection: Callable[[np.ndarray, int], np.ndarray],
    curvature: Callable[[np.ndarray, int], np.ndarray],
    scalar_gradient: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    vector_potential = np.asarray(connection(positions, state), dtype=float)
    velocity = (canonical_momenta + vector_potential) / masses
    force0 = -np.asarray(scalar_gradient(positions, masses, state), dtype=float)
    velocity = velocity + 0.5 * dt * force0 / masses
    velocity = _boris_rotate(velocity, curvature(positions, state), masses, dt)
    predicted = positions + dt * velocity
    force1 = -np.asarray(scalar_gradient(predicted, masses, state), dtype=float)
    velocity = velocity + 0.5 * dt * force1 / masses
    new_positions = positions + dt * velocity
    new_canonical = masses * velocity - np.asarray(connection(new_positions, state), dtype=float)
    return new_positions, new_canonical
