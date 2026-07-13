import os
import sys
from bundle import *
from models import *
from classprop import *
from propagate import PropH          # your exp(-iHdt) in orthonormal frame
from testutils import *
from spawn_ot import *
from modrk45solv import *
import cma

h=1.0e-1
coup_thresh=1.0e-3
nstep = 3000
model = BerryModel2DParallelTransport(0.02,3.0,15.0)

init_rvec = np.asarray([-3.0, 0.0])
init_pvec = np.asarray([20.0, 0.0])

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


# Single-state single-TBF classical propagation
ndim = 2
bundle = Bundle(ndim,model)
init_state = 1
bundle.add_trajectory(init_rvec,init_pvec,init_state)

# Current energies?
for idx in (0, 1):
    print(f"State {idx+1} energy: {model.V_adiabatic(init_rvec,idx)}")

# First, sim till we hit max coupling
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
check_frequency = 50

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

    solverk45_and_step_classical(bundle, h, verbose=False)
    TBF = bundle.trajectorylist[0]
    P_new = TBF.pvec
    r_new = TBF.rvec
 
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

os.makedirs("./saved_zero",exist_ok=True)
np.savetxt("saved_zero/saved_steps.txt",np.asarray(saved_steps))
np.savetxt("saved_zero/saved_times.txt",np.asarray(saved_steps)*h)
np.savetxt("saved_zero/parent_rvecs.txt",np.asarray(parent_rvecs))
np.savetxt("saved_zero/parent_pvecs.txt",np.asarray(parent_pvecs))
np.savetxt("saved_zero/parent_pmechvecs.txt",np.asarray(parent_pmechvecs))
np.savetxt("saved_zero/child_rvecs.txt",np.asarray(child_rvecs))
np.savetxt("saved_zero/child_pvecs.txt",np.asarray(child_pvecs))
np.savetxt("saved_zero/child_pmechvecs.txt",np.asarray(child_pmechvecs))
np.savetxt("saved_zero/coup_mags.txt",np.asarray(coup_mags))
np.savetxt("saved_zero/diff_rvecs.txt",np.asarray(parent_rvecs)-np.asarray(child_rvecs))
np.savetxt("saved_zero/diff_pvecs.txt",np.asarray(parent_pvecs)-np.asarray(child_pvecs))
np.savetxt("saved_zero/diff_pmechvecs.txt",np.asarray(parent_pmechvecs)-np.asarray(child_pmechvecs))
