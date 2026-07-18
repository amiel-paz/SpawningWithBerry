"""The deterministic two-state conical model bundled with PySpawn.

This provider exists as an executable cross-engine reference.  It intentionally
uses PySpawn's row-vector wavefunction convention and phase rule, while exposing
the result through the normal backend-neutral electronic-structure contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..electronic.base import (
    BaseProvider,
    ElectronicProperties,
    ElectronicStructureRequest,
    ElectronicStructureResult,
    ProviderCapabilities,
    WavefunctionState,
)


class PySpawnTestCone(BaseProvider):
    capabilities = ProviderCapabilities(
        energies=True,
        gradients=True,
        state_overlaps=True,
        npi_tdc=True,
    )

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}
        self._counter = 0
        self._checkpoint_references: frozenset[str] | None = None

    def set_checkpoint_references(self, identifiers: frozenset[str]) -> None:
        self._checkpoint_references = identifiers

    def _previous(self, state: WavefunctionState | None) -> dict[str, Any] | None:
        if state is None:
            return None
        if isinstance(state.payload, dict) and "vectors" in state.payload:
            return state.payload
        return self._states.get(state.identifier)

    def evaluate(self, request: ElectronicStructureRequest) -> ElectronicStructureResult:
        self.capabilities.require(request.properties)
        x, y = map(float, request.geometry.reshape(-1)[:2])
        radius = math.hypot(x, y)
        if radius <= 1.0e-14:
            raise ValueError("PySpawn test cone is singular at the origin")
        theta = 0.5 * math.atan2(y, x)

        all_energies = np.asarray((radius * radius - 2.0 * radius,
                                   radius * radius + 2.0 * radius))
        all_gradients = np.zeros((2, len(request.atoms), 3), dtype=float)
        all_gradients[0, 0, :2] = 2.0 * (radius - 1.0) * np.asarray((x, y)) / radius
        all_gradients[1, 0, :2] = 2.0 * (radius + 1.0) * np.asarray((x, y)) / radius

        vectors = np.asarray(
            ((math.sin(theta), math.cos(theta)),
             (math.cos(theta), -math.sin(theta))),
            dtype=float,
        )
        previous = self._previous(request.previous)
        if previous is None:
            overlap = np.eye(2, dtype=np.complex128)
            overlap_dt = 0.0
        else:
            overlap = np.asarray(previous["vectors"]) @ vectors.T
            for state in range(2):
                if overlap[state, state].real < 0.0:
                    vectors[state] *= -1.0
                    overlap[:, state] *= -1.0
            overlap_dt = abs(float(request.time) - float(previous["time"]))

        indices = np.asarray(request.states, dtype=int)
        energies = all_energies[indices]
        gradients = None
        gradient_mask = None
        if ElectronicProperties.GRADIENTS in request.properties:
            gradients = np.zeros((len(indices), len(request.atoms), 3), dtype=float)
            gradient_mask = np.zeros(len(indices), dtype=bool)
            requested = request.gradient_states or request.states
            for state in requested:
                local = request.states.index(state)
                gradients[local] = all_gradients[state]
                gradient_mask[local] = True

        state_overlap = None
        if ElectronicProperties.STATE_OVERLAPS in request.properties:
            state_overlap = overlap[np.ix_(indices, indices)]

        identifier = f"pyspawn-cone-{self._counter:012d}"
        self._counter += 1
        payload = {"vectors": vectors.copy(), "time": float(request.time)}
        self._states[identifier] = payload
        return ElectronicStructureResult(
            energies=energies,
            gradients=gradients,
            gradient_mask=gradient_mask,
            state_overlaps=state_overlap,
            wavefunction=WavefunctionState(identifier, payload),
            metadata={
                "provider": "pyspawn_test_cone",
                "overlap_time_step": overlap_dt,
                "npi_overlap_certified": True,
            },
        ).validate(request)

    def dump_state(self, directory: Path) -> dict[str, Any]:
        states = self._states
        if self._checkpoint_references is not None:
            states = {key: value for key, value in states.items()
                      if key in self._checkpoint_references}
        manifest: dict[str, str] = {}
        for identifier, state in states.items():
            filename = f"{identifier}.npz"
            np.savez_compressed(
                directory / filename,
                vectors=state["vectors"],
                time=np.asarray(state["time"]),
            )
            manifest[identifier] = filename
        (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        return {"manifest": "manifest.json", "counter": self._counter}

    def load_state(self, directory: Path, metadata: dict[str, Any]) -> None:
        self._states.clear()
        self._counter = int(metadata.get("counter", 0))
        manifest_name = metadata.get("manifest")
        if not manifest_name:
            return
        manifest = json.loads((directory / manifest_name).read_text())
        for identifier, filename in manifest.items():
            archive = np.load(directory / filename)
            self._states[identifier] = {
                "vectors": archive["vectors"],
                "time": float(archive["time"]),
            }


def pyspawn_test_cone_factory(_config) -> PySpawnTestCone:
    return PySpawnTestCone()
