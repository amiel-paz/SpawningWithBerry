import numpy as np
from pathlib import Path
from bundle import Bundle
from models import BerryModel2DParallelTransport
from classprop import step_minimal   # your version
from propagate import *          # your exp(-iHdt) in orthonormal frame
from prune import *
from plotter import *
from multidgaussfuncs import *
from math import comb
from random import sample
import random, itertools as it

SEED = 9876543
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
random.seed(SEED)

ndim=2
h=5.0e-2
nstep = 6000
model = BerryModel2DParallelTransport(0.02,3.0,15.0)

thresh = 1.0e-3
frequency = 1
N_samp = 10
N_init = 4
N_last_idx = 20

# 1) Get our initial bundle in there
bundle_orig = Bundle(ndim,model)
init_rvec = np.asarray([-3.0, 0.0])
init_pvec = np.asarray([20.0, 0.0])   # canonical
init_state = 1                        # excited
bundle_orig.add_trajectory(init_rvec, init_pvec, init_state)
TBF_orig = bundle_orig.trajectorylist[0]
G = np.diag(np.reciprocal(TBF_orig.get_masses()))
m = TBF_orig.get_masses()

# 2) Sample from available Gaussian-sampled phase space points
def rand_combos(N, M, m):
    def unrank(k, m, M):
        r, x = [], 1
        for i in range(m, 0, -1):
            for j in range(x, M+1):
                c = comb(M-j, i-1)
                if k < c: r.append(j); x = j+1; break
                k -= c
        return tuple(r)
    return [unrank(k, m, M) for k in sample(range(comb(M, m)), N)]

all_samples = rand_combos(N_samp,N_last_idx,N_init)

# 3) The main loop
for composite_idx in range(N_samp):
    overlaps = np.zeros(N_init, dtype=np.complex128)
    main_bundle = Bundle(ndim,model)
    child_bundle = Bundle(ndim,model)
    child_bundle.add_trajectory(init_rvec, init_pvec, init_state)
    for raw_idx, samp_idx in enumerate(all_samples[composite_idx]):
        folder = Path(f"saved_{samp_idx}/")
        
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
    
        # 4) load sim params
        h_backprop = (saved_times[1]/saved_steps[1]).real
    
        # 5) get our initial bundle in there
        init_rvec = parent_rvecs[0,:]
        init_pvec = parent_pvecs[0,:]   # canonical
        init_state = 1                        # excited
        main_bundle.add_trajectory(init_rvec, init_pvec, init_state)
        TBF_samp = main_bundle.trajectorylist[-1]
        overlaps[raw_idx] = ApplyToTraj(TBF_samp, TBF_orig, OverlapR, False)
        #overlaps[raw_idx] = ApplyToTraj(TBF_orig, TBF_samp, OverlapR, False)
    
        # 6) Diversify, backprop, and prune child bundles
        child_state = 0
    
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
            child_bundle.add_trajectory(r, p, child_state)

        
    child_bundle, kept_child_global_indices = prune_children_by_singularity(
        child_bundle,
        child_state=child_state,
        coup_mags_for_children=coup_mags,
        eig_rel_thresh=1e-4,    # you can tighten/loosen this
        max_cond=1e3,           # target conditioning
        verbose=True
    )

    # Project initial wavefunction to parent TBFs 
    main_bundle.BuildS()       # should be analytic / essentially Hermitian
    main_bundle.BuildSp5inv()  # gives you bundle.Sinv
    main_bundle.BuildH()       # uses KE analytic + PE quadrature
    C = main_bundle.Sinv @ overlaps
    C /= math.sqrt(np.vdot(C, main_bundle.S @ C))
    print(f"Initial norm before addition of child TBFs: {np.vdot(C, main_bundle.S @ C)}")
    main_bundle.SetC(C)

    # Add surviving child bundle TBFs to main bundle
    for idx in range(1,len(child_bundle.trajectorylist)):
        child_TBF = child_bundle.trajectorylist[idx]
        main_bundle.add_trajectory(child_TBF.rvec, child_TBF.pvec, child_TBF.state)
    main_bundle.BuildS()       # should be analytic / essentially Hermitian
    main_bundle.BuildSp5inv()  # gives you bundle.Sinv
    main_bundle.BuildH()       # uses KE analytic + PE quadrature
    C = main_bundle.GetC()
    print(f"Initial norm before addition of child TBFs: {np.vdot(C, main_bundle.S @ C)}")
    
    folder_name = "_".join(map(str, all_samples[composite_idx])) 
    folder = Path(f"saved_{folder_name}/")
    os.makedirs(f"{folder}",exist_ok=True)
    pops_xs = []
    pops_gs = []
    pops_steps = []
    check_every = 20
    nx, ny = 256, 256
    Lx, Ly = 20.0, 20.0
    X, Y   = make_fixed_grid(nx=nx, ny=ny, Lx=Lx, Ly=Ly)
    for step in range(nstep):
        if step % check_every == 0:
            print(f"Step {step}")
            print(f"Excited state population: {np.vdot(main_bundle.C[:N_init], main_bundle.S[:N_init,:N_init] @ main_bundle.C[:N_init]).real}")
            print(f"Ground state population: {np.vdot(main_bundle.C[N_init:], main_bundle.S[N_init:,N_init:] @ main_bundle.C[N_init:]).real}")
            pops_steps.append(step)
            pops_xs.append(np.vdot(main_bundle.C[:N_init], main_bundle.S[:N_init,:N_init] @ main_bundle.C[:N_init]).real)
            pops_gs.append(np.vdot(main_bundle.C[N_init:], main_bundle.S[N_init:,N_init:] @ main_bundle.C[N_init:]).real)
            psi0 = reconstruct_state_on_grid(main_bundle, X, Y, state_index=0)
            psi1 = reconstruct_state_on_grid(main_bundle, X, Y, state_index=1)
            dx = (X.max() - X.min()) / X.shape[1]
            dy = (Y.max() - Y.min()) / Y.shape[0]
            rho0 = (np.abs(psi0)**2) * dx * dy  # convert continuum → discrete normalization
            rho1 = (np.abs(psi1)**2) * dx * dy  # convert continuum → discrete normalization
            save_state_density_frame_fixed(f"{folder}/frames_adiabatic", step, X, Y, rho0, state_index=0)
            save_state_density_frame_fixed(f"{folder}/frames_adiabatic", step, X, Y, rho1, state_index=1)
            np.savetxt(f"{folder}/frame_{step}_GS_pop.txt",rho0)
            np.savetxt(f"{folder}/frame_{step}_XS_pop.txt",rho1)
        advance_one_step(main_bundle, h, nslice=20, move_nuclei=True, renormalize=True)
    
    np.savetxt(f"{folder}/pops_xs.txt",np.asarray(pops_xs))
    np.savetxt(f"{folder}/pops_gs.txt",np.asarray(pops_gs))
    np.savetxt(f"{folder}/pops_steps.txt",np.asarray(pops_steps))
    np.savetxt(f"{folder}/pops_times.txt",np.asarray(pops_steps)*h)
