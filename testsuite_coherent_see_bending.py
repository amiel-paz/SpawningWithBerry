import sys
from bundle import *
from models import *
from classprop import *
from propagate import PropH          # your exp(-iHdt) in orthonormal frame
from testutils import *

h=5.0e-2
coup_thresh=1.0e-3
nstep = 6000
model = BerryModel2DParallelTransport(0.02,3.0,5.0)

init_rvec = np.asarray([-3.0, 0.0])
init_pvec = np.asarray([20.0, 0.0])

# --- Headless plotting (no windows) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from pathlib import Path
import numpy as np
import math

# ---------------- Fixed grid to match your split-operator script ----------------
def make_fixed_grid(nx=256, ny=256, Lx=20.0, Ly=20.0):
    """
    Same lattice as your FFT script:
      x = linspace(-Lx/2, Lx/2 - dx, nx) with dx=Lx/nx (periodic-friendly)
    """
    dx, dy = Lx / nx, Ly / ny
    x = np.linspace(-Lx/2,  Lx/2 - dx, nx)
    y = np.linspace(-Ly/2,  Ly/2 - dy, ny)
    X, Y = np.meshgrid(x, y, indexing='xy')
    return X, Y

# ---------------- Heller FG utilities (ħ=1, no gamma) ----------------
def _fg2d_on_grid(R, P, a, X, Y):
    """
    χ_k(x,y) = N * exp[ -a_x (x-Rx)^2 - a_y (y-Ry)^2 + i * P·(r-R) ].
    Accepts scalar a or 2-vector (ax, ay). Uses Re(a) for normalization.
    """
    Rx, Ry = float(R[0]), float(R[1])
    Px, Py = float(P[0]), float(P[1])

    # allow scalar or per-dim
    if np.ndim(a) == 0:
        ax = ay = complex(a)
    else:
        ax, ay = complex(a[0]), complex(a[1])

    axR, ayR = np.real(ax), np.real(ay)
    if axR <= 0 or ayR <= 0:
        raise ValueError(f"Re(a) must be > 0 for normalization; got Re(ax)={axR}, Re(ay)={ayR}")

    # Normalization for separable product of 1D Gaussians
    N = (2.0 * axR / np.pi) ** 0.25 * (2.0 * ayR / np.pi) ** 0.25

    dx = X - Rx
    dy = Y - Ry
    return N * np.exp(-ax * dx * dx - ay * dy * dy + 1j * (Px * dx + Py * dy))

def reconstruct_excited_adiabatic_on_grid(bundle, X, Y, state_index=1):
    """
    ψ_+(x,y) = sum_{k in state_index} C_k χ_k(x,y), i.e., excited (+A) only.
    Assumes bundle.GetC() aligns with bundle.trajectorylist order.
    """
    C = np.asarray(bundle.GetC(), dtype=np.complex128)
    psi = np.zeros_like(X, dtype=np.complex128)

    for k, traj in enumerate(bundle.trajectorylist):
        if traj.get_state() != state_index:
            continue
        chi = _fg2d_on_grid(traj.rvec, traj.pvec, traj.widths, X, Y)
        psi += C[k] * chi
    return psi

# ---------------- Save a fixed-style frame (no autoscaling) ----------------
def save_excited_density_frame_fixed(outdir, step, X, Y, rho_plus,
                                     title="Excited state density",
                                     fixed_vmax=1e-3):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    extent = [X.min(), X.max(), Y.min(), Y.max()]  # fixed viewport
    plt.figure(figsize=(5, 4), dpi=140)
    plt.imshow(np.real(rho_plus), origin='lower', extent=extent, aspect='auto',
               vmin=0.0, vmax=fixed_vmax)
    plt.colorbar()
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f"{title}  step={step}")
    plt.tight_layout()
    plt.savefig(outdir / f'frame_adiabatic_{step:06d}_1.png')  # trailing "_1" matches your “tag” style
    plt.close()

# ---------------- Driver that plugs into your propagation ----------------
def run_with_fixed_excited_frames(bundle,
                                  advance_one_step_fn,
                                  nstep,
                                  dt,
                                  X, Y,
                                  frame_stride=200,
                                  outdir="sim_out/frames_adiabatic",
                                  nslice=20,
                                  renormalize_physical=True,
                                  move_nuclei=True,
                                  state_index=1,
                                  fixed_vmax=1e-3):
    """
    Propagate your TBF bundle and dump |ψ_+(x,y)|^2 every `frame_stride` steps.
    Uses a fixed grid & fixed color scale like your FFT script.
    """
    for step in range(1, nstep + 1):
        # Advance one step of your coupled coeff + classical nuclei
        _, phys_norm = advance_one_step_fn(bundle, dt, nslice=nslice, move_nuclei=move_nuclei)

        # Optional physical renormalization to keep coefficients tidy
        if renormalize_physical and phys_norm > 0:
            C_new = bundle.GetC() / math.sqrt(phys_norm)
            bundle.SetC(C_new)

        print(f"Current QM energy = {bundle.E_expec()}")
        # Save a frame on stride
        if step % frame_stride == 0:
            psi_plus = reconstruct_excited_adiabatic_on_grid(bundle, X, Y, state_index=state_index)
            rho_plus = np.abs(psi_plus) ** 2
            dx = (X.max() - X.min()) / X.shape[1]
            dy = (Y.max() - Y.min()) / Y.shape[0]
            rho_plot = (np.abs(psi_plus)**2) * dx * dy  # convert continuum → discrete normalization
            save_excited_density_frame_fixed(outdir, step, X, Y, rho_plot,
                                             title="Excited state density", fixed_vmax=fixed_vmax)


def advance_one_step(bundle, dt, nslice=20, move_nuclei=True):
    """
    - build Heff = H - i Sdot
    - propagate coefficients with exp(-i Heff dt)
    - optionally move nuclei classically
    returns (C_new, phys_norm)
    """
    model = bundle.model

    # build everything once
    bundle.BuildHeff()

    C_t = np.asarray(bundle.GetC(), dtype=np.complex128)

    # coeffs in non-orthonormal basis, but PropH already handles Heff
    C_new = PropH(bundle.Heff, C_t, dt, nslice, renormalize=False)
    bundle.SetC(C_new)

    if move_nuclei:
        for traj in bundle.trajectorylist:
            r_k = traj.get_rvec()
            P_k = traj.get_pvec()
            st  = traj.get_state()
            m_k = traj.get_masses()

            r_next, P_next = step_minimal(
                r_k, P_k, m_k, st,
                model.A, model.Omega, model.gradU,
                dt, hbar=1.0
            )
            traj.set_rvec(r_next)
            traj.set_pvec(P_next)

    # physical norm
    bundle.BuildS()
    C_new = bundle.GetC()
    phys_norm = np.vdot(C_new, bundle.S @ C_new).real
    return C_new, phys_norm

# Single-state single-TBF classical propagation
ndim = 2
bundle = Bundle(ndim,model)
init_state = 1
bundle.add_trajectory(init_rvec,init_pvec,init_state)
TBF = bundle.trajectorylist[0]
rng = np.random.default_rng(1234)

# Current energies?
for idx in (0, 1):
    print(f"State {idx+1} energy: {model.V_adiabatic(init_rvec,idx)}")

#bundle, E_proj = projected_refined_energy_MC(TBF, model, 8, include_original=True, rng=rng, Eclass_deviation=0.15)
print(f"Starting QM energy = {bundle.E_expec()}")

# --- fixed viewport to match fft_dynamics.py defaults ---
nx, ny = 256, 256
Lx, Ly = 20.0, 20.0
X, Y   = make_fixed_grid(nx=nx, ny=ny, Lx=Lx, Ly=Ly)

print("Starting simple sim on excited state (fixed viewport like FFT script)")

run_with_fixed_excited_frames(
    bundle=bundle,
    advance_one_step_fn=advance_one_step,
    nstep=nstep,
    dt=h,
    X=X, Y=Y,
    frame_stride=200,                          # same as your bash runner
    outdir="sim_out/frames_adiabatic",         # same folder naming vibe
    nslice=20,
    renormalize_physical=True,
    move_nuclei=True,
    state_index=1,                             # excited (+A) only
    fixed_vmax=1e-3                            # fixed color scale like your example
)

