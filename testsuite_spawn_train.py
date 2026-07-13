import numpy as np
from pathlib import Path
from bundle import Bundle
from models import BerryModel2DParallelTransport
from classprop import step_minimal   # your version
from propagate import *          # your exp(-iHdt) in orthonormal frame
from prune import *
from plotter import *

ndim=2
h=5.0e-2
nstep = 5000
model = BerryModel2DParallelTransport(0.02,3.0,5.0)

thresh = 1.0e-3
folder = Path("saved_zero/")
frequency = 1

# 1) indices where coup_mags > thresh
coup_mags = np.loadtxt(folder / "coup_mags.txt", dtype=np.complex128).real
idx = np.flatnonzero(coup_mags > thresh)

# 2) helper that loads as complex and masks rows
def load_masked(name):
    arr = np.loadtxt(folder / name, dtype=np.complex128)
    arr = np.atleast_1d(arr)
    if arr.ndim == 1:
        return arr[idx]
    else:
        return arr[idx, ...]

# 3) grab masked arrays
child_pmechvecs   = load_masked("child_pmechvecs.txt")[::frequency]
diff_rvecs        = load_masked("diff_rvecs.txt")[::frequency]
parent_rvecs      = load_masked("parent_rvecs.txt")[::frequency]
child_pvecs       = load_masked("child_pvecs.txt")[::frequency]          # complex-safe
diff_pmechvecs    = load_masked("diff_pmechvecs.txt")[::frequency]
parent_pmechvecs  = load_masked("parent_pmechvecs.txt")[::frequency]
saved_steps       = load_masked("saved_steps.txt")[::frequency]
child_rvecs       = load_masked("child_rvecs.txt")[::frequency]
diff_pvecs        = load_masked("diff_pvecs.txt")[::frequency]           # complex-safe
parent_pvecs      = load_masked("parent_pvecs.txt")[::frequency]         # complex-safe
saved_times       = load_masked("saved_times.txt")[::frequency]
coup_mags         = load_masked("coup_mags.txt")[::frequency]

masked_indices = idx  # keep if you need original row numbers
print(coup_mags)

# 4) load sim params
h_backprop = (saved_times[1]/saved_steps[1]).real

# 5) get our initial bundle in there
bundle = Bundle(ndim,model)
init_rvec = np.asarray([-3.0, 0.0])
init_pvec = np.asarray([20.0, 0.0])   # canonical
init_state = 1                        # excited
bundle.add_trajectory(init_rvec, init_pvec, init_state)
TBF = bundle.trajectorylist[0]
G = np.diag(np.reciprocal(TBF.get_masses()))
m = TBF.get_masses()

child_state = 0

"""
for idx in range(len(coup_mags)):
    r = child_rvecs[idx]
    p = child_pvecs[idx]
    print(int(saved_steps[idx]))
    for step in range(int(saved_steps[idx])):
        r, p = step_minimal(r, p, m, child_state,
                                    model.A, model.Omega, model.gradU, -h_backprop)
    print(f"Adding a a TBF at {r}, {p}")
    bundle.add_trajectory(r, p, child_state)
"""

seed_idx = select_diverse_by_time_and_geometry(
    child_rvecs, child_pvecs, coup_mags, saved_steps,
    K_per_bucket=1, bucket_width=100, r_tol=0.15, p_tol=0.5
)

for j in seed_idx:
    r = child_rvecs[j].real
    p = child_pvecs[j].real
    # backprop with the exact forward step h
    for _ in range(int(saved_steps[j])):
        r, p = step_minimal(r, p, m, child_state, model.A, model.Omega, model.gradU, -h)
    bundle.add_trajectory(r, p, child_state)

bundle, kept_child_global_indices = prune_children_by_singularity(
    bundle,
    child_state=child_state,
    coup_mags_for_children=coup_mags,
    eig_rel_thresh=1e-8,    # you can tighten/loosen this
    max_cond=1e4,           # target conditioning
    verbose=True
)

pops_xs = []
pops_gs = []
pops_steps = []
bundle.BuildS()
print(bundle.S)
check_every = 20
nx, ny = 256, 256
Lx, Ly = 20.0, 20.0
X, Y   = make_fixed_grid(nx=nx, ny=ny, Lx=Lx, Ly=Ly)
for step in range(nstep):
    if step % check_every == 0:
        print(f"Step {step}")
        print(f"Excited state population: {abs(bundle.C[0])**2}")
        print(f"Ground state population: {np.vdot(bundle.C[1:], bundle.S[1:,1:] @ bundle.C[1:]).real}")
        pops_steps.append(step)
        pops_xs.append(abs(bundle.C[0])**2)
        pops_gs.append(np.vdot(bundle.C[1:], bundle.S[1:,1:] @ bundle.C[1:]).real)
        psi0 = reconstruct_state_on_grid(bundle, X, Y, state_index=0)
        psi1 = reconstruct_state_on_grid(bundle, X, Y, state_index=1)
        dx = (X.max() - X.min()) / X.shape[1]
        dy = (Y.max() - Y.min()) / Y.shape[0]
        rho0 = (np.abs(psi0)**2) * dx * dy  # convert continuum → discrete normalization
        rho1 = (np.abs(psi1)**2) * dx * dy  # convert continuum → discrete normalization
        save_state_density_frame_fixed("saved_zero/frames_adiabatic", step, X, Y, rho0, state_index=0)
        save_state_density_frame_fixed("saved_zero/frames_adiabatic", step, X, Y, rho1, state_index=1)
    advance_one_step(bundle, h, nslice=20, move_nuclei=True, renormalize=True)

os.makedirs("./saved_zero",exist_ok=True)
np.savetxt("saved_zero/pops_xs.txt",np.asarray(pops_xs))
np.savetxt("saved_zero/pops_gs.txt",np.asarray(pops_gs))
np.savetxt("saved_zero/pops_steps.txt",np.asarray(pops_steps))
np.savetxt("saved_zero/pops_times.txt",np.asarray(pops_steps)*h)
