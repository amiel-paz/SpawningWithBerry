"""Minimal calculator demonstrating the backend-neutral callable API."""

import numpy as np

from aims_berry import CallableProvider, ProviderCapabilities


def calculate(request):
    coordinate = request.geometry[1, 0] - request.geometry[0, 0]
    energies = np.array([0.01 * coordinate**2, 0.03 + 0.005 * coordinate**2])
    gradients = np.zeros((2, 2, 3))
    gradients[:, 0, 0] = -np.array([0.02, 0.01]) * coordinate
    gradients[:, 1, 0] = -gradients[:, 0, 0]
    nacs = np.zeros((2, 2, 2, 3), dtype=complex)
    nacs[0, 1, 0, 0] = 0.02
    nacs[0, 1, 1, 0] = -0.02
    nacs[1, 0] = -nacs[0, 1].conj()
    return {"energies": energies, "gradients": gradients, "nacs": nacs}


def make_provider(_config):
    return CallableProvider(
        calculate,
        capabilities=ProviderCapabilities(energies=True, gradients=True, nacs=True),
    )
