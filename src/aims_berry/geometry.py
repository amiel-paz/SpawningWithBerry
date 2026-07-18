"""Geometry, masses, widths, and initial-condition helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import BOHR_PER_ANGSTROM, ConfigError

ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}
ATOMIC_MASSES_AMU = {
    "H": 1.00784, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998403163, "S": 32.06, "Cl": 35.45,
}
GAUSSIAN_WIDTHS = {"H": 6.0, "C": 30.0, "N": 22.0, "O": 22.0, "F": 22.0, "S": 18.0, "Cl": 18.0}
AMU_TO_ELECTRON_MASS = 1822.888486209


def read_xyz(path: str | Path, units: str = "angstrom") -> tuple[tuple[str, ...], np.ndarray]:
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    try:
        natom = int(lines[0])
    except (IndexError, ValueError) as exc:
        raise ConfigError(f"{path}: invalid XYZ atom count") from exc
    rows = lines[2:2 + natom]
    if len(rows) != natom:
        raise ConfigError(f"{path}: expected {natom} atom rows")
    atoms: list[str] = []
    coords: list[list[float]] = []
    for row in rows:
        fields = row.split()
        if len(fields) < 4:
            raise ConfigError(f"{path}: invalid XYZ row {row!r}")
        atoms.append(fields[0])
        coords.append(list(map(float, fields[1:4])))
    xyz = np.asarray(coords, dtype=float)
    if units == "angstrom":
        xyz *= BOHR_PER_ANGSTROM
    elif units != "bohr":
        raise ConfigError(f"unsupported geometry_units {units!r}")
    return tuple(atoms), xyz


def atomic_numbers(atoms: tuple[str, ...]) -> np.ndarray:
    try:
        return np.asarray([ATOMIC_NUMBERS[a] for a in atoms], dtype=int)
    except KeyError as exc:
        raise ConfigError(f"unsupported element {exc.args[0]!r}; supply a custom provider") from exc


def masses_and_widths(
    atoms: tuple[str, ...],
    explicit_widths: tuple[float, ...] = (),
    explicit_masses: tuple[float, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    try:
        masses = np.repeat([ATOMIC_MASSES_AMU[a] * AMU_TO_ELECTRON_MASS for a in atoms], 3)
        default_widths = np.repeat([GAUSSIAN_WIDTHS[a] for a in atoms], 3)
    except KeyError as exc:
        raise ConfigError(f"no mass/width default for element {exc.args[0]!r}") from exc
    if explicit_masses:
        masses = np.asarray(explicit_masses, dtype=float)
        if masses.size not in (len(atoms), 3 * len(atoms)):
            raise ConfigError("nuclear_masses must have natom or 3*natom entries")
        if masses.size == len(atoms):
            masses = np.repeat(masses, 3)
    if explicit_widths:
        widths = np.asarray(explicit_widths, dtype=float)
        if widths.size not in (len(atoms), 3 * len(atoms)):
            raise ConfigError("gaussian_widths must have natom or 3*natom entries")
        if widths.size == len(atoms):
            widths = np.repeat(widths, 3)
    else:
        widths = default_widths
    if np.any(widths <= 0):
        raise ConfigError("Gaussian widths must be positive")
    if np.any(~np.isfinite(masses)) or np.any(masses <= 0):
        raise ConfigError("nuclear masses must be positive and finite")
    return masses.astype(float), widths.astype(float)


def read_momenta(path: str | Path, natom: int) -> np.ndarray:
    values: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) == 4:
            fields = fields[1:]
        if len(fields) != 3:
            raise ConfigError(f"{path}: expected three momentum components per line")
        values.append(list(map(float, fields)))
    if len(values) != natom:
        raise ConfigError(f"{path}: expected {natom} momentum rows")
    return np.asarray(values, dtype=float)


def sample_wigner(
    geometry: np.ndarray,
    masses_flat: np.ndarray,
    hessian_path: str | Path,
    rng: np.random.Generator,
    temperature: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the harmonic Wigner distribution from an unweighted Hessian in a text file."""
    n = geometry.size
    raw = np.loadtxt(hessian_path, dtype=float)
    hessian = raw.reshape(n, n)
    inv_sqrt_m = 1.0 / np.sqrt(masses_flat)
    mw_hessian = hessian * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]
    omega2, modes = np.linalg.eigh(0.5 * (mw_hessian + mw_hessian.T))
    keep = omega2 > 1.0e-10
    omega = np.sqrt(omega2[keep])
    q_sigma = 1.0 / np.sqrt(2.0 * omega)
    p_sigma = np.sqrt(omega / 2.0)
    if temperature > 0:
        kelvin_per_hartree = 315775.02480407
        beta = kelvin_per_hartree / temperature
        coth = 1.0 / np.tanh(0.5 * beta * omega)
        q_sigma *= np.sqrt(coth)
        p_sigma *= np.sqrt(coth)
    q = rng.normal(size=omega.size) * q_sigma
    p = rng.normal(size=omega.size) * p_sigma
    cart_modes = inv_sqrt_m[:, None] * modes[:, keep]
    displacement = cart_modes @ q
    momentum = np.sqrt(masses_flat) * (modes[:, keep] @ p)
    return (geometry.reshape(-1) + displacement).reshape(geometry.shape), momentum.reshape(geometry.shape)
