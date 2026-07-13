"""Adapter for small, home-built electronic calculators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import (
    BaseProvider,
    ElectronicStructureRequest,
    ElectronicStructureResult,
    ProviderCapabilities,
)


class CallableProvider(BaseProvider):
    def __init__(
        self,
        function: Callable[[ElectronicStructureRequest], ElectronicStructureResult | dict[str, Any]],
        *,
        capabilities: ProviderCapabilities | None = None,
    ) -> None:
        self.function = function
        self.capabilities = capabilities or ProviderCapabilities(nacs=True, complex_values=True)

    def evaluate(self, request: ElectronicStructureRequest) -> ElectronicStructureResult:
        self.capabilities.require(request.properties)
        result = self.function(request)
        if isinstance(result, dict):
            result = ElectronicStructureResult(**result)
        if not isinstance(result, ElectronicStructureResult):
            raise TypeError("calculator must return ElectronicStructureResult or a compatible dict")
        return result.validate(request)
