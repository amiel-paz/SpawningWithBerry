"""Typed parser for the flat, space-delimited AIMS input format."""

from __future__ import annotations

import dataclasses
import shlex
from pathlib import Path
from typing import Any, Literal

AU_TIME_PER_FS = 41.34137333518211
BOHR_PER_ANGSTROM = 1.8897261254578281


class ConfigError(ValueError):
    """Raised for a configuration error with source context."""


@dataclasses.dataclass(frozen=True)
class ObservableSpec:
    kind: Literal["bond", "angle", "dihedral"]
    name: str
    atoms: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class SimulationConfig:
    provider: str
    geometry: Path
    geometry_units: Literal["angstrom", "bohr"] = "angstrom"
    num_states: int = 1
    initial_state: int = 0
    time_step: float = 1.0
    coupling_time_step: float | None = None
    simulation_time: float = 100.0
    random_seed: int = 0
    initial_condition: Literal["direct", "file", "wigner"] = "file"
    momenta: Path | None = None
    hessian: Path | None = None
    temperature: float = 0.0
    electronic_method: str = "sa_casscf"
    scf_method: str = "rhf"
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0
    active_electrons: int | None = None
    active_orbitals: int | None = None
    state_weights: tuple[float, ...] = ()
    coupling_mode: Literal["auto", "npi", "nac"] = "auto"
    quantum_integrator: Literal["cayley", "rk45"] = "cayley"
    spawn_strategy: Literal["nac", "coupling_optimized"] = "nac"
    spawn_momentum: Literal["nac", "isotropic"] = "nac"
    spawn_metric: Literal["projected", "nac_norm", "tdc"] = "projected"
    spawn_threshold: float = 0.01
    population_to_spawn: float = 1.0e-3
    spawn_overlap_max: float = 0.8
    spawn_cooldown: float = 0.0
    max_trajectories: int = 100
    max_energy_gap: float = float("inf")
    nac_gap_threshold: float = float("inf")
    pair_overlap_threshold: float = 0.0
    overlap_threshold: float = 1.0e-3
    regularization_threshold: float = 1.0e-8
    min_time_step: float | None = None
    minimum_nuclear_time_step: float | None = None
    energy_tolerance: float = 5.0e-3
    quantum_energy_policy: Literal["record", "error"] = "record"
    classical_energy_policy: Literal["record", "error"] = "error"
    adaptive_classical_timestep: bool = True
    classical_energy_tolerance: float | None = None
    classical_energy_numerical_margin: float = 0.0
    norm_tolerance: float = 1.0e-6
    cumulative_norm_tolerance: float = 1.0e-8
    output_every: int = 1
    write_xyz: bool = False
    checkpoint_keep: int = 2
    electronic_retries: int = 2
    run_directory: Path = Path("run")
    gaussian_widths: tuple[float, ...] = ()
    nuclear_masses: tuple[float, ...] = ()
    provider_options: tuple[tuple[str, Any], ...] = ()
    observables: tuple[ObservableSpec, ...] = ()
    source: Path | None = None

    def __post_init__(self) -> None:
        if self.num_states < 1:
            raise ConfigError("num_states must be positive")
        if not 0 <= self.initial_state < self.num_states:
            raise ConfigError("initial_state must be in [0, num_states)")
        if self.time_step <= 0 or self.simulation_time < 0:
            raise ConfigError("time_step must be positive and simulation_time non-negative")
        if self.coupling_time_step is not None and not 0 < self.coupling_time_step <= self.time_step:
            raise ConfigError("coupling_time_step must be in (0, time_step]")
        if self.min_time_step is not None and not 0 < self.min_time_step <= self.time_step:
            raise ConfigError("min_time_step must be in (0, time_step]")
        if self.minimum_nuclear_time_step is not None:
            upper = self.coupling_time_step or self.time_step
            if not 0 < self.minimum_nuclear_time_step <= upper:
                raise ConfigError("minimum_nuclear_time_step must be in (0, coupling_time_step]")
        if len(self.state_weights) not in (0, self.num_states):
            raise ConfigError("state_weights must contain num_states entries")
        if self.state_weights and not abs(sum(self.state_weights) - 1.0) < 1.0e-10:
            raise ConfigError("state_weights must sum to one")
        if self.active_electrons is None and self.electronic_method == "sa_casscf":
            raise ConfigError("active_electrons is required for sa_casscf")
        if self.active_orbitals is None and self.electronic_method == "sa_casscf":
            raise ConfigError("active_orbitals is required for sa_casscf")
        if self.initial_condition == "wigner" and self.hessian is None:
            raise ConfigError("hessian is required for Wigner initial conditions")
        if self.initial_condition not in {"direct", "file", "wigner"}:
            raise ConfigError("initial_condition must be direct, file, or wigner")
        if self.coupling_mode not in {"auto", "npi", "nac"}:
            raise ConfigError("coupling_mode must be auto, npi, or nac")
        if self.quantum_integrator not in {"cayley", "rk45"}:
            raise ConfigError("quantum_integrator must be cayley or rk45")
        if self.quantum_energy_policy not in {"record", "error"}:
            raise ConfigError("quantum_energy_policy must be record or error")
        if self.classical_energy_policy not in {"record", "error"}:
            raise ConfigError("classical_energy_policy must be record or error")
        if self.spawn_strategy not in {"nac", "coupling_optimized"}:
            raise ConfigError("spawn_strategy must be nac or coupling_optimized")
        if self.spawn_momentum not in {"nac", "isotropic"}:
            raise ConfigError("spawn_momentum must be nac or isotropic")
        if self.spawn_metric not in {"projected", "nac_norm", "tdc"}:
            raise ConfigError("spawn_metric must be projected, nac_norm, or tdc")
        if self.coupling_mode == "nac" and self.spawn_metric == "tdc":
            raise ConfigError("spawn_metric tdc requires coupling_mode auto or npi")
        if self.nac_gap_threshold <= 0:
            raise ConfigError("nac_gap_threshold must be positive")
        if (
            self.classical_energy_tolerance is not None
            and self.classical_energy_tolerance <= 0
        ):
            raise ConfigError("classical_energy_tolerance must be positive")
        if self.classical_energy_numerical_margin < 0:
            raise ConfigError("classical_energy_numerical_margin must be non-negative")
        if self.norm_tolerance <= 0:
            raise ConfigError("norm_tolerance must be positive")
        if self.cumulative_norm_tolerance <= 0:
            raise ConfigError("cumulative_norm_tolerance must be positive")
        if self.output_every < 1:
            raise ConfigError("output_every must be positive")
        if self.pair_overlap_threshold < 0 or self.pair_overlap_threshold >= 1:
            raise ConfigError("pair_overlap_threshold must be in [0, 1)")

    @property
    def nsteps(self) -> int:
        return int(round(self.simulation_time / self.time_step))

    def provider_option_dict(self) -> dict[str, Any]:
        return dict(self.provider_options)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        for key in ("geometry", "momenta", "hessian", "run_directory", "source"):
            if data[key] is not None:
                data[key] = str(data[key])
        return data


_SCALAR_TYPES: dict[str, type] = {
    "provider": str,
    "geometry": Path,
    "geometry_units": str,
    "num_states": int,
    "initial_state": int,
    "random_seed": int,
    "initial_condition": str,
    "momenta": Path,
    "hessian": Path,
    "temperature": float,
    "electronic_method": str,
    "scf_method": str,
    "basis": str,
    "charge": int,
    "spin": int,
    "active_electrons": int,
    "active_orbitals": int,
    "coupling_mode": str,
    "quantum_integrator": str,
    "spawn_strategy": str,
    "spawn_momentum": str,
    "spawn_metric": str,
    "spawn_threshold": float,
    "population_to_spawn": float,
    "spawn_overlap_max": float,
    "spawn_cooldown": float,
    "max_trajectories": int,
    "max_energy_gap": float,
    "nac_gap_threshold": float,
    "pair_overlap_threshold": float,
    "overlap_threshold": float,
    "regularization_threshold": float,
    "energy_tolerance": float,
    "quantum_energy_policy": str,
    "classical_energy_policy": str,
    "adaptive_classical_timestep": bool,
    "classical_energy_tolerance": float,
    "classical_energy_numerical_margin": float,
    "norm_tolerance": float,
    "cumulative_norm_tolerance": float,
    "output_every": int,
    "write_xyz": bool,
    "checkpoint_keep": int,
    "electronic_retries": int,
    "run_directory": Path,
}
_TIME_KEYS = {
    "time_step", "coupling_time_step", "simulation_time", "min_time_step",
    "minimum_nuclear_time_step",
}
_LIST_KEYS = {"state_weights", "gaussian_widths", "nuclear_masses"}
_REQUIRED = {"provider", "geometry", "num_states", "initial_state", "time_step", "simulation_time"}


def _atom_value(text: str) -> Any:
    lower = text.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _time_value(tokens: list[str]) -> float:
    if not tokens or len(tokens) > 2:
        raise ValueError("expected VALUE [au|fs]")
    value = float(tokens[0])
    unit = tokens[1].lower() if len(tokens) == 2 else "au"
    if unit in {"au", "a.u."}:
        return value
    if unit in {"fs", "femtosecond", "femtoseconds"}:
        return value * AU_TIME_PER_FS
    raise ValueError(f"unsupported time unit {unit!r}")


def load_config(path: str | Path) -> SimulationConfig:
    source = Path(path).expanduser().resolve()
    raw: dict[str, Any] = {}
    provider_options: list[tuple[str, Any]] = []
    observables: list[ObservableSpec] = []
    for lineno, line in enumerate(source.read_text().splitlines(), 1):
        try:
            tokens = shlex.split(line, comments=True, posix=True)
        except ValueError as exc:
            raise ConfigError(f"{source}:{lineno}: {exc}") from exc
        if not tokens:
            continue
        key, values = tokens[0], tokens[1:]
        try:
            if key == "provider_option":
                if len(values) < 2:
                    raise ValueError("provider_option requires NAME VALUE")
                value: Any = _atom_value(values[1]) if len(values) == 2 else tuple(_atom_value(v) for v in values[1:])
                provider_options.append((values[0], value))
                continue
            if key == "observable":
                if len(values) < 4:
                    raise ValueError("observable requires KIND NAME ATOM...")
                kind, name = values[:2]
                expected = {"bond": 2, "angle": 3, "dihedral": 4}.get(kind)
                if expected is None or len(values[2:]) != expected:
                    raise ValueError("observable must be bond/angle/dihedral with 2/3/4 atoms")
                observables.append(ObservableSpec(kind, name, tuple(map(int, values[2:]))))
                continue
            if key in raw:
                raise ValueError(f"duplicate key {key!r}")
            if key in _TIME_KEYS:
                raw[key] = _time_value(values)
            elif key in _LIST_KEYS:
                if not values:
                    raise ValueError("expected one or more numeric values")
                raw[key] = tuple(map(float, values))
            elif key in _SCALAR_TYPES:
                if len(values) != 1:
                    raise ValueError("expected exactly one value")
                typ = _SCALAR_TYPES[key]
                if typ is bool:
                    value = _atom_value(values[0])
                    if not isinstance(value, bool):
                        raise ValueError("expected true/false, yes/no, or on/off")
                    raw[key] = value
                else:
                    raw[key] = typ(values[0])
            else:
                raise ValueError(f"unknown keyword {key!r}")
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{source}:{lineno}: {exc}") from exc
    missing = sorted(_REQUIRED - raw.keys())
    if missing:
        raise ConfigError(f"{source}: missing required keywords: {', '.join(missing)}")
    base = source.parent
    for key in ("geometry", "momenta", "hessian", "run_directory"):
        if key in raw and raw[key] is not None and not raw[key].is_absolute():
            raw[key] = (base / raw[key]).resolve()
    raw["provider_options"] = tuple(provider_options)
    raw["observables"] = tuple(observables)
    raw["source"] = source
    return SimulationConfig(**raw)


def config_from_dict(data: dict[str, Any]) -> SimulationConfig:
    values = dict(data)
    for key in ("geometry", "momenta", "hessian", "run_directory", "source"):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    values["state_weights"] = tuple(values.get("state_weights", ()))
    values["gaussian_widths"] = tuple(values.get("gaussian_widths", ()))
    values["nuclear_masses"] = tuple(values.get("nuclear_masses", ()))
    values["provider_options"] = tuple(tuple(item) for item in values.get("provider_options", ()))
    values["observables"] = tuple(ObservableSpec(item["kind"], item["name"], tuple(item["atoms"])) for item in values.get("observables", ()))
    return SimulationConfig(**values)
