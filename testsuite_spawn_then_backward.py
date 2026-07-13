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
model = BerryModel2DParallelTransport(0.02,3.0,5.0)

init_rvec = np.asarray([-3.0, 0.1])
init_pvec = np.asarray([20.0, -0.2])

def run_cma_spawn(obj, verbose=True):
    x0 = np.array([0.0, 0.0, 0.0])
    best_x = x0.copy()
    best_f = 1e30

    sigma0 = 0.05
    pop = 30
    n_restart = 3

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

# Single-state single-TBF classical propagation
ndim = 2
bundle = Bundle(ndim,model)
init_state = 1
bundle.add_trajectory(init_rvec,init_pvec,init_state)

# Current energies?
for idx in (0, 1):
    print(f"State {idx+1} energy: {model.V_adiabatic(init_rvec,idx)}")

# First, sim till we hit max coupling
print("Starting simple sim on excited state")
ntraj = bundle.ntraj
max_coup_mag = None
max_step = None
for step in range(nstep):
    TBF = bundle.trajectorylist[0]
    r = TBF.rvec
    P = TBF.pvec               # canonical
    m = TBF.get_masses()
    state = TBF.state
 
    # Check momentum of a directly cloned trajectory
    temp_state = 1 - state
    E_parent = model.classical_energy(r,P,m,state)
    Phi_parent = (
        model.V_adiabatic(r, state)
        + model.D_off(r, m, state)
        + model.div_d(r, m, state, state)
    ).real
    u_dir = P / np.linalg.norm(P)
    P_child = energy_matched_canonical_momentum(r, u_dir, model, m, E_parent, 
                                                   Phi_parent, temp_state)
    if (P_child is not None): 
        temp_bundle = copy.deepcopy(bundle)
        temp_bundle.add_trajectory(r, P_child, temp_state)
        temp_bundle.BuildHeff()
        C_test = PropH(temp_bundle.Heff, temp_bundle.C, h, nslice=20, renormalize=False)
        nS = np.vdot(C_test, temp_bundle.S @ C_test).real
        C_test /= np.sqrt(nS)
        curr_coup_mag = abs(C_test[1])
        print(f"Current coupling magnitude: {curr_coup_mag}")
        if (curr_coup_mag > coup_thresh):
            if (max_coup_mag is not None and curr_coup_mag < max_coup_mag):
                print(f"Coupling suddenly decreased. Time to spawn and backpropagate") 
                break
            max_coup_mag = curr_coup_mag
            max_step = step+1

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

if (max_step == None): 
    print(f"Coupling never broke threshold, so no backpropagation will happen.")
    sys.exit()

parent_traj = bundle.trajectorylist[0]
mass = TBF.get_masses()
R_p = parent_traj.get_rvec()
P_p = parent_traj.get_pvec()
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

if P_child is None:
    print("[spawn] energy match failed, skipping child")
    sys.exit()

# 2) insert child with zero amplitude
child_idx = insert_child(bundle,
                         R_child,
                         P_child,
                         target_state,
                         amp=0.0+0.0j)

print(f"[spawn] inserted child at idx={child_idx} on state {target_state}, R={R_child}, P={P_child}")

print(f"Now, attempt to propagate back to initial conditions ({max_step} steps)")
ntraj = bundle.ntraj
for step in range(max_step):
    for traj in range(ntraj):
        TBF = bundle.trajectorylist[traj]
        r = TBF.rvec
        P = TBF.pvec               # canonical
        m = TBF.get_masses()
        state = TBF.state
        r_new, P_new = step_minimal(r, P, m, state,
                                    model.A, model.Omega, model.gradU, -h)
        bundle.trajectorylist[traj].set_rvec(r_new)
        bundle.trajectorylist[traj].set_pvec(P_new)
     
        A_new = model.A(r_new, state)
        p_mech = P_new + A_new     # hbar = 1
        print(f"traj {traj} on state {state}")
        print(f"step {step}  r = {r_new}")
        print(f"canonical P = {P_new}")
        print(f"mechanical p = {p_mech}")
 
        Eclass = model.classical_energy(r_new, P_new, m, state)
        print(f"classical energy = {Eclass}")

print(f"Now, forward propagate with both TBFs")
for step in range(nstep):
    advance_one_step(bundle, h, nslice=20, move_nuclei=True)
    bundle.BuildHeff()
    H = bundle.H
    C = bundle.GetC()
    print(f"Current populations: {np.square(np.abs(C))}")
    print(f"Current energy: {np.vdot(C, H @ C).real}")
    for traj in range(ntraj):
        TBF = bundle.trajectorylist[traj]
        r = TBF.rvec
        P = TBF.pvec               # canonical
        m = TBF.get_masses()
        state = TBF.state
     
        A = model.A(r, state)
        p_mech = P + A     # hbar = 1
        print(f"traj {traj} on state {state}")
        print(f"step {step}  r = {r}")
        print(f"canonical P = {P}")
        print(f"mechanical p = {p_mech}")

        Eclass = model.classical_energy(r, P, m, state)
        print(f"classical energy = {Eclass}")
