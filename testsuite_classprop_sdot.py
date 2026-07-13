from bundle import *
from models import *
from classprop import *
from testutils import *
from modrk45solv import *

# Single-state single-TBF classical propagation
ndim = 2
factor = 50
nstep = int(30000 / factor)
dt_main=1.0e-2 * factor
dt_fd     = 1.0e-4   # small FD timestep (e.g. 0.01)
model = BerryModel2DParallelTransport(0.02,3.0,5.0)
bundle = Bundle(ndim,model)

#init_rvec = np.asarray([-3.0, 0.0])
init_rvec = np.asarray([0.0, 0.0])
init_pvec = np.asarray([20.0, 0.0])
init_state = 1
bundle.add_trajectory(init_rvec,init_pvec,init_state)

# Current energies?
for idx in (0, 1):
    print(f"State {idx+1} energy: {model.V_adiabatic(init_rvec,idx)}")

# Current overlap is unit?
bundle.BuildS()
print(bundle.S)

#Current rdot and pdot?
TBF = bundle.trajectorylist[0]
G = np.diag(np.reciprocal(TBF.get_masses()))
m = TBF.get_masses()
print(model.get_rdot(G,init_rvec,init_pvec,init_state))
print(model.get_pdot(G,init_rvec,init_pvec,init_state))

# Current Sdot?
bundle.BuildSDot()

# Current overlap?
bundle.BuildS()

# Numerical Sdot?
print("Starting simple sim on excited state")
ntraj = bundle.ntraj
state = 1
# Start with a *single*-trajectory bundle
for step in range(nstep):
    # 1) Propagate main trajectory by dt_main
    solverk45_and_step_classical(bundle, dt_main, verbose=False)

    # 2) Snapshot current state (this is t_k)
    base = copy.deepcopy(bundle)
    T0   = base.trajectorylist[0]
    r0   = T0.rvec.copy()
    p0   = T0.pvec.copy()
    m    = T0.get_masses()
    state = T0.state

    # 3) Build FD clones at t_k
    back = copy.deepcopy(base)
    fwd  = copy.deepcopy(base)

    # backward/forward small steps
    solverk45_and_step_classical(back, -dt_fd, verbose=False)
    solverk45_and_step_classical(fwd,  +dt_fd, verbose=False)

    # 4) Create a temporary 3-trajectory bundle to reuse your S-builder
    temp = Bundle(ndim, model)
    T_main = base.trajectorylist[0]
    T_b    = back.trajectorylist[0]
    T_f    = fwd.trajectorylist[0]

    temp.add_trajectory(T_main.rvec, T_main.pvec, T_main.state)
    temp.add_trajectory(T_b.rvec,    T_b.pvec,    T_b.state)
    temp.add_trajectory(T_f.rvec,    T_f.pvec,    T_f.state)

    temp.BuildS()

    # 5) Analytic Sdot at t_k from *single*-traj bundle
    base.BuildSDot()
    realSdot = base.SDot[0, 0]

    # 6) FD Sdot at t_k
    S = temp.S
    fdSdot = 1j * ((S[0, 2] - S[0, 1]) / (2 * dt_fd)).imag

    # 7) Print diagnostics
    print(f"step {step}  r = {r0}")
    print(f"canonical P = {p0}")
    A_new = model.A(r0, state)
    p_mech = p0 + A_new
    print(f"mechanical p = {p_mech}")
    Eclass = model.classical_energy(r0, p0, m, state)
    print(f"classical energy = {Eclass}")

    print(f"Computed Sdot:          {realSdot}")
    print(f"Finite difference Sdot: {fdSdot}")
    print(f"Deviation:              {abs(realSdot - fdSdot)}")
