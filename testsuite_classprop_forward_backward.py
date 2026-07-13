from bundle import *
from models import *
from classprop import *
from testutils import *

# Single-state single-TBF classical propagation
ndim = 2
h=1.0e-2
nstep = 20000
model = BerryModel2DParallelTransport(0.02,3.0,5.0)
bundle = Bundle(ndim,model)

init_rvec = np.asarray([-3.0, 0.0])
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

# Current KE?
print(bundle.BuildKE())

# Current PE?
print(bundle.BuildPE())

# Current Sdot?
bundle.BuildSDot()
print(bundle.SDot)

# Generate backward displacement
bundle.add_trajectory(init_rvec, init_pvec, init_state)
r_new,p_new = step_minimal(init_rvec, init_pvec, m, init_state, model.A, model.Omega, model.gradU, -h)
bundle.trajectorylist[1].set_rvec(r_new)
bundle.trajectorylist[1].set_pvec(p_new)

# Generate forward displacement
bundle.add_trajectory(init_rvec, init_pvec, init_state)
r_new,p_new = step_minimal(init_rvec, init_pvec, m, init_state, model.A, model.Omega, model.gradU, h)
bundle.trajectorylist[2].set_rvec(r_new)
bundle.trajectorylist[2].set_pvec(p_new)

# Current overlap?
bundle.BuildS()

# Numerical Sdot?
print("Starting simple sim on excited state")
ntraj = bundle.ntraj
for step in range(nstep):
    for traj in range(ntraj):
        TBF = bundle.trajectorylist[traj]
        r = TBF.rvec
        P = TBF.pvec               # canonical
        state = TBF.state

        r_new, P_new = step_minimal(r, P, m, state,
                                    model.A, model.Omega, model.gradU, h)
        bundle.trajectorylist[traj].set_rvec(r_new)
        bundle.trajectorylist[traj].set_pvec(P_new)

        if traj == 0:
            A_new = model.A(r_new, state)
            p_mech = P_new + A_new     # hbar = 1
            print(f"step {step}  r = {r_new}")
            print(f"canonical P = {P_new}")
            print(f"mechanical p = {p_mech}")

            Eclass = model.classical_energy(r_new, P_new, m, state)
            print(f"classical energy = {Eclass}")

print("Now going backwards")
for step in range(nstep):
    for traj in range(ntraj):
        TBF = bundle.trajectorylist[traj]
        r = TBF.rvec
        P = TBF.pvec               # canonical
        state = TBF.state

        r_new, P_new = step_minimal(r, P, m, state,
                                    model.A, model.Omega, model.gradU, -h)
        bundle.trajectorylist[traj].set_rvec(r_new)
        bundle.trajectorylist[traj].set_pvec(P_new)

        if traj == 0:
            A_new = model.A(r_new, state)
            p_mech = P_new + A_new     # hbar = 1
            print(f"step {step}  r = {r_new}")
            print(f"canonical P = {P_new}")
            print(f"mechanical p = {p_mech}")

            Eclass = model.classical_energy(r_new, P_new, m, state)
            print(f"classical energy = {Eclass}")
