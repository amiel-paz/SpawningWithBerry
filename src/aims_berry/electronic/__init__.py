"""Backend-neutral electronic-structure contracts and adapters."""

from .base import (
    BaseProvider,
    ElectronicProperties,
    ElectronicStructureError,
    ElectronicStructureProvider,
    ElectronicStructureRequest,
    ElectronicStructureResult,
    ProviderCapabilities,
    ProviderCapabilityError,
    WavefunctionState,
)
from .callable import CallableProvider

__all__ = [
    "BaseProvider", "CallableProvider", "ElectronicProperties", "ElectronicStructureError",
    "ElectronicStructureProvider", "ElectronicStructureRequest", "ElectronicStructureResult",
    "ProviderCapabilities", "ProviderCapabilityError", "WavefunctionState",
]
