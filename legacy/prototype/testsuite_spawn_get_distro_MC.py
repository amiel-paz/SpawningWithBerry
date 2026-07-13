import os
import sys
from bundle import *
from models import *
from classprop import *
from propagate import PropH          # your exp(-iHdt) in orthonormal frame
from testutils import *
from spawn_ot import *
import cma

h=5.0e-2
coup_thresh=1.0e-3
nstep = 5000
model = BerryModel2DParallelTransport(0.02,3.0,15.0)

init_rvec = np.asarray([-3.0, 0.0])
init_pvec = np.asarray([20.0, 0.0])
sigma_R = 1.0 / (2.0 * np.sqrt(np.ones(2)))
sigma_P = np.sqrt(np.ones(2))
N_samples = 50

def run_cma_spawn(obj, verbose=True):
    x0 = np.array([0.0, 0.0, 0.0])
    best_x = x0.copy()
    best_f = 1e30

    sigma0 = 0.02
    pop = 30
    n_restart = 2

    for r in range(n_restart):
        opts = {
            'verb_disp': 1 if verbose else 0,
            'maxiter': 200,
            'maxfevals': 8000,
            'tolfun': 1e-9,
            'tolfunhist': 1e-9,
            'tolx': 1e-7,
            'popsize': pop,
            'bounds': [[-0.2, -0.2, -np.pi],
                       [ 0.2,  0.2,  np.pi]],
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        while not es.stop():
            X = es.ask()
            F = []
            for x in X:
                fval = obj(np.asarray(x))
                F.append(fval)
                if fval < best_f:
                    best_f = fval
                    best_x = np.asarray(x)
            es.tell(X, F)
            if verbose and es.countiter % 5 == 0:
                print(f"[CMA r={r}] it={es.countiter}, fbest={best_f:.6e}, xbest={best_x}")

        if verbose:
            print(f"[CMA r={r}] done, best_f={best_f:.6e}, best_x={best_x}")

        # prep next restart
        pop = int(pop * 1.6)
        sigma0 *= 0.6

    return best_x, best_f

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


# Loop to generate
rng = np.random.default_rng()
init_state = 1
ndim = 2
check_frequency = 50
for samp_idx in range(N_samples):

    # Sample a new phase space point
    dR = rng.normal(loc=0.0, scale=sigma_R, size=ndim)
    dP = rng.normal(loc=0.0, scale=sigma_P, size=ndim)

    # Single-state single-TBF classical propagation
    bundle = Bundle(ndim,model)
    bundle.add_trajectory(init_rvec+dR,init_pvec+dP,init_state)

    coup_mags = []
    saved_steps = []
    parent_rvecs = []
    parent_pvecs = []
    parent_pmechvecs = []
    child_rvecs = []
    child_pvecs = []
    child_pmechvecs = []
    print("Starting simple sim on excited state")
    ntraj = bundle.ntraj
    
    for step in range(nstep):
        TBF = bundle.trajectorylist[0]
        r = TBF.rvec
        P = TBF.pvec               # canonical
        m = TBF.get_masses()
        state = TBF.state
    
        if step % check_frequency == 0:
            parent_traj = bundle.trajectorylist[0]
            R_p = parent_traj.get_rvec()
            P_p = parent_traj.get_pvec()
            mass = TBF.get_masses()
            curr_state = parent_traj.state
            target_state = 1 - curr_state
            
            # 1) make objective tied to THIS parent
            obj = make_spawn_objective_exact(bundle.GetC()[0], parent_traj, model, h, target_state)
            
            best_x, best_f = run_cma_spawn(obj)
            
            # decode best_x → R_child, P_child
            dx, dy, theta = best_x
            R_p = parent_traj.get_rvec()
            R_child = R_p + np.array([dx, dy], float)
            u_dir   = np.array([np.cos(theta), np.sin(theta)], float)
            
            # energy-match momentum
            E_parent = model.classical_energy(R_p,
                                              P_p,
                                              mass,
                                              curr_state)
            
            Phi_parent = (
                model.V_adiabatic(R_p, curr_state)
                + model.D_off(R_p, mass, curr_state)
                + model.div_d(R_p, mass, curr_state, curr_state)
            ).real
            
            P_child = energy_matched_canonical_momentum(
                R_child,
                u_dir,
                model,
                mass,
                E_parent,
                Phi_parent,
                target_state
            )
            A_child = model.A(R_child, target_state)
            P_mech_child = P_child + A_child
            A_parent = model.A(R_p, curr_state)
            P_mech_parent = P_p + A_parent
    
            temp_bundle = copy.deepcopy(bundle)
            insert_child(temp_bundle, R_child, P_child, target_state)
            temp_bundle.BuildHeff()
            coup_mags.append(abs(temp_bundle.Heff[0,1]))  
            saved_steps.append(step) 
            parent_rvecs.append(R_p)
            parent_pvecs.append(P_p)
            parent_pmechvecs.append(P_mech_parent)
            child_rvecs.append(R_child)
            child_pvecs.append(P_child)
            child_pmechvecs.append(P_mech_child)
    
        r_new, P_new = step_minimal(r, P, m, state,
                                    model.A, model.Omega, model.gradU, h)
        bundle.trajectorylist[0].set_rvec(r_new)
        bundle.trajectorylist[0].set_pvec(P_new)
     
        A_new = model.A(r_new, state)
        p_mech = P_new + A_new     # hbar = 1
        print(f"step {step}  r = {r_new}")
        print(f"canonical P = {P_new}")
        print(f"mechanical p = {p_mech}")
     
        Eclass = model.classical_energy(r_new, P_new, m, state)
        print(f"classical energy = {Eclass}")
    
    for idx in range(len(child_rvecs)):
        print(f"Step {saved_steps[idx]}:")
        print(f"Parent {parent_rvecs[idx]}{parent_pmechvecs[idx]} spawns")
        print(f"Child {child_rvecs[idx]}{child_pmechvecs[idx]}")
        print(f"With magnitude {coup_mags[idx]}")
    
    os.makedirs(f"./saved_{samp_idx+1}",exist_ok=True)
    np.savetxt(f"saved_{samp_idx+1}/saved_steps.txt",np.asarray(saved_steps))
    np.savetxt(f"saved_{samp_idx+1}/saved_times.txt",np.asarray(saved_steps)*h)
    np.savetxt(f"saved_{samp_idx+1}/parent_rvecs.txt",np.asarray(parent_rvecs))
    np.savetxt(f"saved_{samp_idx+1}/parent_pvecs.txt",np.asarray(parent_pvecs))
    np.savetxt(f"saved_{samp_idx+1}/parent_pmechvecs.txt",np.asarray(parent_pmechvecs))
    np.savetxt(f"saved_{samp_idx+1}/child_rvecs.txt",np.asarray(child_rvecs))
    np.savetxt(f"saved_{samp_idx+1}/child_pvecs.txt",np.asarray(child_pvecs))
    np.savetxt(f"saved_{samp_idx+1}/child_pmechvecs.txt",np.asarray(child_pmechvecs))
    np.savetxt(f"saved_{samp_idx+1}/coup_mags.txt",np.asarray(coup_mags))
    np.savetxt(f"saved_{samp_idx+1}/diff_rvecs.txt",np.asarray(parent_rvecs)-np.asarray(child_rvecs))
    np.savetxt(f"saved_{samp_idx+1}/diff_pvecs.txt",np.asarray(parent_pvecs)-np.asarray(child_pvecs))
    np.savetxt(f"saved_{samp_idx+1}/diff_pmechvecs.txt",np.asarray(parent_pmechvecs)-np.asarray(child_pmechvecs))
