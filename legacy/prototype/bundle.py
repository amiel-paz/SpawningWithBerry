import os, math
import numpy as np
import matplotlib.pyplot as plt
from math import erf, pi, sqrt, exp
from scipy import special
from trajectory import Trajectory
import onedgauss
from complexint import *
from multidgaussfuncs import *
from gaussfuncs import *

class Bundle(object):

    def __init__(self, ndim, model):
        self.ndim = ndim
        self.ntraj = 0
        self.model = model
        self.trajectorylist = []

        # Quantities needed for TDSE solution in time-dependent
        # non-orthogonal basis
        self.S = None
        self.H = None
        self.Heff = None
        self.Sp5inv = None
        self.Sinv = None
        self.SDot = None
        self.C = []

        # Safeguards for spawning
        self.spawn_meta = {}

    def add_trajectory(self, rvec, pvec, state):
        new_traj = Trajectory(self.ndim)
        new_traj.set_rvec(rvec)
        new_traj.set_pvec(pvec)
        new_traj.set_state(state) 
        self.trajectorylist.append(new_traj)
        if len(self.C) == 0:
            self.C = np.append(np.asarray(self.C), 1)
        else:
            self.C = np.append(np.asarray(self.C), 0)
        self.spawn_meta[len(self.trajectorylist)] = {
                "last_spawn_step": -np.inf,
                "last_spawn_target": None,
                }
        self.ntraj = len(self.trajectorylist)

    def E_expec(self):
        self.BuildH()
        return (np.vdot(self.C, self.H @ self.C)).real

    def BuildS(self):
        self.S = np.zeros((self.ntraj,self.ntraj),dtype=np.complex128)
        for I in range(self.ntraj):
            TBF_I = self.trajectorylist[I]
            for J in range(I,self.ntraj):
                TBF_J = self.trajectorylist[J]
                self.S[I,J] = (TBF_I.state == TBF_J.state) * ApplyToTraj(TBF_I,TBF_J,OverlapR,False)
                self.S[J,I] = self.S[I,J].conjugate()

    def BuildSp5inv(self, thresh=1.0e-4):
        self.Sinv = np.zeros((self.ntraj,self.ntraj),dtype=np.complex128)
        self.Sp5inv = np.zeros((self.ntraj,self.ntraj),dtype=np.complex128)
        Ctemp = np.zeros((self.ntraj,self.ntraj),dtype=np.complex128)
        CtempSp5inv = np.zeros((self.ntraj,self.ntraj),dtype=np.complex128)
        self.BuildS()
        S_Eval,S_Evec = np.linalg.eigh(self.S)
        for I in range(self.ntraj):
            Dstemp = 0
            DstempSp5inv = 0
            if (abs(S_Eval[I]) > thresh):
                Dstemp = 1.0 / S_Eval[I]
                DstempSp5inv = 1.0 / math.sqrt(S_Eval[I])
            else:
                continue
            Ctemp[I,:] = Dstemp * S_Evec[:,I].conjugate()
            CtempSp5inv[I,:] = DstempSp5inv * S_Evec[:,I].conjugate()
        self.Sinv = S_Evec @ Ctemp
        self.Sp5inv = S_Evec @ CtempSp5inv

    def BuildSDot(self): 
        self.SDot = np.zeros((self.ntraj,self.ntraj),dtype=np.complex128)
        for I in range(self.ntraj):
            TBF_I = self.trajectorylist[I]
            for J in range(self.ntraj):
                TBF_J = self.trajectorylist[J]
                if (TBF_I.state == TBF_J.state):
                    rvec = TBF_J.get_rvec()
                    pvec = TBF_J.get_pvec()
                    G = np.diag(np.reciprocal(TBF_J.get_masses()))
                    rdot_vec = self.model.get_rdot(G,rvec,pvec,TBF_J.state) 
                    pdot_vec = self.model.get_pdot(G,rvec,pvec,TBF_J.state)
                    adot_vec = self.model.get_adot(G,rvec,pvec,TBF_J.state)
                    gammadot_vec = self.model.get_gammadot(G,rvec,pvec,TBF_J.state)
                    S_list = ApplyToIndividual(TBF_I,TBF_J,OverlapR,False)
                    SDot_list = SDotTrajIndividual(TBF_I,TBF_J,rdot_vec,pdot_vec,adot_vec,gammadot_vec,False)
                    for K in range(self.ndim):
                        S_prod_exclude_K = np.prod([S_list[L] for L in range(len(S_list)) if L != K])
                        self.SDot[I,J] += SDot_list[K] * S_prod_exclude_K

          
    def BuildKE(self):
        KE = np.zeros((self.ntraj,self.ntraj),dtype=np.complex128)
        for I in range(self.ntraj):
            TBF_I = self.trajectorylist[I]
            for J in range(I,self.ntraj):
                TBF_J = self.trajectorylist[J]
                if (TBF_I.state != TBF_J.state):
                    continue
                S_list = ApplyToIndividual(TBF_I,TBF_J,OverlapR,False)
                KE_list = ApplyToIndividual(TBF_I,TBF_J,Kinetic,False)
                for K in range(self.ndim):
                    S_prod_exclude_K = np.prod([S_list[L] for L in range(len(S_list)) if L != K])
                    KE[I,J] += KE_list[K] * S_prod_exclude_K
                KE[J,I] = KE[I,J].conjugate() 
        return KE

    def BuildPE_slow(self):
        """
        Build the full PE matrix with proper multidimensional quadrature
        for *position-dependent* gauge terms.
    
        PE_ij =
            ∫ χ_i^*(R) χ_j(R) [ V(R) + D(R) + div_d(R) + ... ] dR
          + ∑_alpha ∫ χ_i^*(R) [ (-i/m_alpha) d_alpha(R) ∂_alpha χ_j(R) ] dR
    
        where each scalar term is provided via a callable.
        """
    
        PE = np.zeros((self.ntraj, self.ntraj), dtype=np.complex128)
    
        for I in range(self.ntraj):
            TBF_I = self.trajectorylist[I]
    
            for J in range(I, self.ntraj):
                TBF_J = self.trajectorylist[J]
    
                bra_state = TBF_I.state
                ket_state = TBF_J.state
                mass_vec  = np.asarray(TBF_J.get_masses(), dtype=float)
                ndim      = TBF_I.ndim
    
                PE_IJ = 0.0 + 0j
    
                # ---- scalar terms ----
                scalar_terms = [
                    self.scalar_V,
                    self.scalar_D,
                    self.scalar_divd,
                    # drop more callables in here whenever you invent them
                ]
    
                for scalar_fn in scalar_terms:
                    f_scalar = self.integrand_scalar(
                        TBF_I, TBF_J,
                        self.model,
                        scalar_fn
                    )
                    PE_IJ += integrate_nd(f_scalar, ndim)
    
                # ---- momentum · derivative-coupling term ----
                mom_sum = 0.0 + 0j
                for alpha in range(ndim):
                    f_alpha = self.integrand_momentum_dim(
                        TBF_I, TBF_J,
                        self.model,
                        alpha
                    )
                    mom_sum += integrate_nd(f_alpha, ndim)
    
                PE_IJ += mom_sum
    
                # Hermitian fill
                PE[I,J] = PE_IJ
                PE[J,I] = PE_IJ.conjugate()
    
        return PE

    def BuildPE_2D_fast(self,
                     Nx=64, Ny=64, L=6.0,
                     overlap_cutoff=1e-12):
        """
        Fast potential/H_berry builder for the bundle using tensor-product
        Gauss–Legendre quadrature over truncated domains.
    
        self must have:
            self.ntraj
            self.trajectorylist  (list of TBFs)
            self.model           (the BerryModel2DParallelTransport)
    
        Returns:
            PE : (ntraj, ntraj) complex Hermitian matrix
        """
    
        nT = self.ntraj
        PE = np.zeros((nT, nT), dtype=np.complex128)
    
        for I in range(nT):
            TBF_I = self.trajectorylist[I]
            for J in range(I, nT):
                TBF_J = self.trajectorylist[J]
    
                PE_IJ = self.compute_PE_pair_fast(
                    TBF_I, TBF_J, self.model,
                    Nx=Nx, Ny=Ny, L=L,
                    overlap_cutoff=overlap_cutoff
                )
    
                PE[I, J] = PE_IJ
                PE[J, I] = np.conjugate(PE_IJ)
    
        return PE

    def BuildPE(self,
                     Nx=64, Ny=64, L=6.0,
                     overlap_cutoff=1e-12):
        """
        Faster potential/H_berry builder for the bundle using half analytic,
        half Gauss–Hermite quadrature over truncated domains.
    
        self must have:
            self.ntraj
            self.trajectorylist  (list of TBFs)
            self.model           (the BerryModel2DParallelTransport)
    
        Returns:
            PE : (ntraj, ntraj) complex Hermitian matrix
        """
    
        nT = self.ntraj
        PE = np.zeros((nT, nT), dtype=np.complex128)
    
        for I in range(nT):
            TBF_I = self.trajectorylist[I]
            for J in range(I, nT):
                TBF_J = self.trajectorylist[J]
    
                PE_IJ = self.model.eval_integrals(TBF_I,TBF_J,thresh=1.0e-10)
    
                PE[I, J] = PE_IJ
                PE[J, I] = np.conjugate(PE_IJ)
    
        return PE

    def BuildPE_SPA(self):
        PE = np.zeros((self.ntraj,self.ntraj),dtype=np.complex128)
        for I in range(self.ntraj):
            TBF_I = self.trajectorylist[I]
            for J in range(I,self.ntraj):
                TBF_J = self.trajectorylist[J]
                rcent_vec = []
                for n in range(TBF_I.ndim):
                    a1 = TBF_I.gausslist[n].a
                    R1 = TBF_I.gausslist[n].R
                    a2 = TBF_J.gausslist[n].a
                    R2 = TBF_J.gausslist[n].R
                    rcent_vec.append((a1 * R1 + a2 * R2)/(a1 + a2))
                m = TBF_J.get_masses()
                if (TBF_I.state == TBF_J.state):
                    # Add potential
                    PE[I,J] = self.model.V_adiabatic(rcent_vec,TBF_I.state) * ApplyToTraj(TBF_I,TBF_J,OverlapR,False)
                # Add Born-Oppenheimer corrections
                PE[I,J] += self.model.D(rcent_vec,m,TBF_I.state,TBF_J.state) * ApplyToTraj(TBF_I,TBF_J,OverlapR,False)
                PE[I,J] += self.model.div_d(rcent_vec,m,TBF_I.state,TBF_J.state) * ApplyToTraj(TBF_I,TBF_J,OverlapR,False)
                # Add Momentum couplings
                S_list = ApplyToIndividual(TBF_I,TBF_J,OverlapR,False)
                P_list = ApplyToIndividual(TBF_I,TBF_J,Momentum,False)
                d_vec = self.model.d(rcent_vec,TBF_I.state,TBF_J.state)
                p_vec = []
                for K in range(self.ndim):
                    S_prod_exclude_K = np.prod([S_list[L] for L in range(len(S_list)) if L != K])
                    p_vec.append(P_list[K] * S_prod_exclude_K)
                PE[I,J] += -1j * np.dot(d_vec * (1/m),np.asarray(p_vec))
                PE[J,I] = PE[I,J].conjugate() 
        return PE

    def BuildH(self):
        self.H = np.zeros((self.ntraj,self.ntraj),dtype=np.complex128)
        self.H = self.BuildPE() + self.BuildKE()

    def BuildHeff(self):
        self.BuildH()
        self.BuildSDot()
        self.BuildSp5inv()
        self.Heff = self.Sinv @ (self.H - 1j * self.SDot)

    def GetC(self):
        return np.asarray(self.C, dtype=np.complex128)

    def SetC(self, C):
        self.C = C

    # The super bespoke helpers for testing
   
    def squeeze_traj_for_eps(self, traj, eps):
        """
        Make a *copy* of traj but scaled for semiclassical parameter eps.
        This defines the chi_eps packet:
        exp( -(a/eps)*(x-R)^2 + i*(p/eps)*(x-R) )

        Important:
        - We do NOT change traj.rvec or traj.pvec (those are the physical centers R0, P0)
        - But we DO change each underlying 1D Gaussian basis object so that:
          a_eff = a / eps
          p_eff = p / eps
        """

        squeezed = Trajectory(traj.ndim)
        squeezed.set_state(traj.state)

        # copy masses, centers, momenta as physical values
        squeezed.set_rvec(traj.rvec.copy())
        squeezed.set_pvec(traj.pvec.copy())
        squeezed.set_masses(traj.masses.copy())
        squeezed.set_widths(traj.widths.copy())  # we'll override inside gausslist though

        squeezed.gausslist = []
        for k in range(traj.ndim):
            Rk = traj.gausslist[k].R      # center position
            pk = traj.gausslist[k].P      # physical momentum center (this is pvec[k])
            ak = traj.gausslist[k].a      # original width
            mk = traj.gausslist[k].m      # mass

            a_eff = ak / eps
            p_eff = pk / eps   # phase slope in the exponential

            # NOTE: you need OneD_Gauss to let you construct with arbitrary a and p
            squeezed.gausslist.append(
                onedgauss.OneD_Gauss(Rk, p_eff, a_eff, mk, 0.0, 1.0)
            )
        return squeezed

    def Kinetic_eps(self, gI, gJ, eps):
        # rescale the operator: -(eps^2 / 2m) d^2/dR^2
        base_val = Kinetic(gI, gJ, False)     # old value with ħ=1 convention
        return (eps**2) * base_val

    def Momentum_eps(self, gI, gJ, eps):
        # old Momentum() assumed ħ=1, i.e. operator -i d/dR
        base_val = Momentum(gI, gJ, False)   # returns <gI| -i d/dR |gJ>
        return eps * base_val

    def BuildKE_eps(self, eps):
        KE = np.zeros((self.ntraj,self.ntraj),dtype=np.complex128)
    
        for I in range(self.ntraj):
            TBF_I_phys = self.trajectorylist[I]
            TBF_I_sqz  = self.squeeze_traj_for_eps(TBF_I_phys, eps)
    
            for J in range(I,self.ntraj):
                TBF_J_phys = self.trajectorylist[J]
                if (TBF_I_phys.state != TBF_J_phys.state):
                    continue
                TBF_J_sqz  = self.squeeze_traj_for_eps(TBF_J_phys, eps)
    
                # Overlaps in configuration space with squeezed Gaussians
                S_list = ApplyToIndividual(TBF_I_sqz, TBF_J_sqz, OverlapR, False)
    
                # Kinetic matrix elements with ε-scaled operator:
                #    - (eps^2 / 2m) d^2/dR^2
                KE_list = ApplyToIndividual(TBF_I_sqz, TBF_J_sqz,
                                            lambda gI,gJ,_flag: self.Kinetic_eps(gI,gJ,eps),
                                            False)
    
                val = 0.0+0.0j
                for K in range(self.ndim):
                    S_prod_exclude_K = np.prod([S_list[L] for L in range(len(S_list)) if L != K])
                    val += KE_list[K] * S_prod_exclude_K
    
                KE[I,J] = val
                KE[J,I] = val.conjugate()
    
        return KE

    def BuildPE_eps(self,
                    eps,
                    Nx=64, Ny=64, L=6.0,
                    overlap_cutoff=1e-12):
        """
        Fast(ish) PE builder for the eps-scaled (squeezed) bundle.

        This is the eps-analog of BuildPE():
          - nuclear Gaussians are first squeezed with squeeze_traj_for_eps(eps)
          - scalar terms (V + D + div_d) are integrated by quadrature
            rather than saddle-point
          - the momentum-coupling term -i d·(p/m) uses eps-rescaled
            derivative overlaps, matching Momentum_eps.

        Returns:
            PE : (ntraj, ntraj) complex Hermitian matrix
        """

        nT = self.ntraj
        PE = np.zeros((nT, nT), dtype=np.complex128)

        for I in range(nT):
            TBF_I_phys = self.trajectorylist[I]
            for J in range(I, nT):
                TBF_J_phys = self.trajectorylist[J]

                PE_IJ = self.compute_PE_pair_fast_eps(
                    TBF_I_phys,
                    TBF_J_phys,
                    self.model,
                    eps,
                    Nx=Nx,
                    Ny=Ny,
                    L=L,
                    overlap_cutoff=overlap_cutoff
                )

                PE[I, J] = PE_IJ
                PE[J, I] = np.conjugate(PE_IJ)

        return PE


    def BuildPE_SPA_eps(self, eps):
        PE = np.zeros((self.ntraj,self.ntraj),dtype=np.complex128)
    
        for I in range(self.ntraj):
            TBF_I_phys = self.trajectorylist[I]
            TBF_I_sqz  = self.squeeze_traj_for_eps(TBF_I_phys, eps)
    
            for J in range(I,self.ntraj):
                TBF_J_phys = self.trajectorylist[J]
                TBF_J_sqz  = self.squeeze_traj_for_eps(TBF_J_phys, eps)
    
                # 1. effective "center" position for BO potentials etc.
                rcent_vec = []
                for n in range(TBF_I_phys.ndim):
                    a1 = TBF_I_sqz.gausslist[n].a
                    R1 = TBF_I_sqz.gausslist[n].R
                    a2 = TBF_J_sqz.gausslist[n].a
                    R2 = TBF_J_sqz.gausslist[n].R
                    rcent_vec.append((a1 * R1 + a2 * R2)/(a1 + a2))
    
                m_vec = TBF_J_phys.get_masses()
                state_I = TBF_I_phys.state
                state_J = TBF_J_phys.state
    
                # overlap between squeezed packets
                S_overlap = ApplyToTraj(TBF_I_sqz, TBF_J_sqz, OverlapR, False)
    
                val = 0.0+0.0j
    
                # (a) BO potential only if same adiabatic state
                if state_I == state_J:
                    val += self.model.V_adiabatic(rcent_vec, state_I) * S_overlap
    
                # (b) geometric scalar potentials U = D + div_d
                val += self.model.D(rcent_vec, m_vec, state_I, state_J)     * S_overlap
                val += self.model.div_d(rcent_vec, m_vec, state_I, state_J) * S_overlap
    
                # (c) momentum-coupling term: -i d · (p/m)
                # build p_vec using ε-scaled momentum operator
                S_list = ApplyToIndividual(TBF_I_sqz, TBF_J_sqz, OverlapR, False)
                P_list = ApplyToIndividual(TBF_I_sqz, TBF_J_sqz,
                                           lambda gI,gJ,_flag: self.Momentum_eps(gI,gJ,eps),
                                           False)
    
                d_vec = self.model.d(rcent_vec, state_I, state_J)
    
                p_vec = []
                for K in range(self.ndim):
                    S_prod_exclude_K = np.prod([S_list[L] for L in range(len(S_list)) if L != K])
                    p_vec.append(P_list[K] * S_prod_exclude_K)
    
                p_vec = np.asarray(p_vec)
    
                val += (-1j) * np.dot(d_vec * (1/m_vec), p_vec)
    
                PE[I,J] = val
                PE[J,I] = val.conjugate()
    
        return PE

    def integrand_overlap(self, TBF_I, TBF_J):
        def f(R):
            amp = 1.0 + 0j
            for k in range(TBF_I.ndim):
                gI = TBF_I.gausslist[k]
                gJ = TBF_J.gausslist[k]
                amp *= np.conjugate(gI.get_values(R[k])) * gJ.get_values(R[k])
            return amp
        return f

    def integrand_momentum_dim(self, TBF_I, TBF_J, model, alpha):
        def f(R):
            mass_vec  = TBF_I.get_masses()
            bra_state = TBF_I.state
            ket_state = TBF_J.state
    
            # wavefunctions
            gI = TBF_I.gausslist[alpha]
            gJ = TBF_J.gausslist[alpha]
    
            # build χ_I* χ_J and their derivatives along alpha
            amp_bra = 1.0 + 0j
            amp_ket = 1.0 + 0j
            for k in range(TBF_I.ndim):
                GI = TBF_I.gausslist[k]
                GJ = TBF_J.gausslist[k]
                if k == alpha:
                    amp_bra *= np.conjugate(GI.get_dvalues_dx(R[k])) * GJ.get_values(R[k])
                    amp_ket *= np.conjugate(GI.get_values(R[k])) * GJ.get_dvalues_dx(R[k])
                else:
                    valI = GI.get_values(R[k])
                    valJ = GJ.get_values(R[k])
                    amp_bra *= np.conjugate(valI) * valJ
                    amp_ket *= np.conjugate(valI) * valJ
    
            d_vec  = model.d(R, bra_state, ket_state)
            d_a    = d_vec[alpha]
            d_a_cc = np.conjugate(d_a)
    
            # hermitian (anti-symmetric) combo:
            #  (-i/2m) [ d · (χ* ∂χ) - (∂χ*) · d χ ]
            return (-1j / (2.0 * mass_vec[alpha])) * (d_a * amp_ket - d_a_cc * amp_bra)
    
        return f


    def integrand_scalar(
        self,
        TBF_I,
        TBF_J,
        model,
        scalar_func
    ):
        """
        Returns f(R) = χ_I^*(R) χ_J(R) * scalar_func(R, model, mass_vec, bra_state, ket_state)
    
        scalar_func is user-supplied, so this wrapper is generic.
        """
        mass_vec = TBF_I.get_masses()
        bra_state = TBF_I.state
        ket_state = TBF_J.state
        overlap_fn = self.integrand_overlap(TBF_I, TBF_J)
    
        def f(R):
            chi_prod = overlap_fn(R)
            val = scalar_func(R, model, mass_vec, bra_state, ket_state)
            return chi_prod * val
        return f

 
    def scalar_V(self, R, model, mass_vec, bra_state, ket_state):
        # only add Born-Oppenheimer potential on the diagonal state
        if bra_state == ket_state:
            return model.V_adiabatic(R, bra_state)
        else:
            return 0.0
    
    def scalar_D(self, R, model, mass_vec, bra_state, ket_state):
        # geometric D term (DBOC-like)
        return model.D(R, mass_vec, bra_state, ket_state)
    
    def scalar_divd(self, R, model, mass_vec, bra_state, ket_state):
        # divergence of derivative coupling
        return model.div_d(R, mass_vec, bra_state, ket_state)

    def overlap_window_1d(self, gI, gJ, L=20.0):
        """
        Given two 1D Gaussians gI, gJ with attributes
            g.R (center), g.a (real width > 0),
        estimate the region where gI*(x) gJ(x) has support.
    
        Returns:
            R_eff : float   effective center of the product
            x_min : float
            x_max : float
        """
        aI = gI.a
        aJ = gJ.a
        RI = gI.R
        RJ = gJ.R
    
        A_eff = aI + aJ              # effective total width param
        R_eff = (aI*RI + aJ*RJ)/A_eff # center of overlap product
    
        sigma = 1.0 / np.sqrt(2.0 * A_eff)  # std dev of overlap Gaussian
    
        x_min = R_eff - L*sigma
        x_max = R_eff + L*sigma
        return R_eff, x_min, x_max

    def precompute_gaussian_1d(self, gI, gJ, x_nodes):
        """
        Precompute nuclear 1D factors along a given quadrature line.
    
        gI, gJ : OneD_Gauss instances
        x_nodes: (Nx,) array of coordinates along that axis
    
        Returns:
            phiI_conj_x   : (Nx,)  ψ_I^*(x)
            phiJ_x        : (Nx,)  ψ_J(x)
            dphiJdx_x     : (Nx,)  ∂x ψ_J(x)
        """
        Nx = len(x_nodes)
        phiI_conj_x = np.zeros(Nx, dtype=np.complex128)
        phiJ_x      = np.zeros(Nx, dtype=np.complex128)
        dphiJdx_x   = np.zeros(Nx, dtype=np.complex128)
    
        for idx, x in enumerate(x_nodes):
            psiI = gI.get_values(x)
            psiJ = gJ.get_values(x)
            dpsiJdx = gJ.get_dvalues_dx(x)
    
            phiI_conj_x[idx] = np.conjugate(psiI)
            phiJ_x[idx]      = psiJ
            dphiJdx_x[idx]   = dpsiJdx
    
        return phiI_conj_x, phiJ_x, dphiJdx_x

    def compute_PE_pair_fast(self, TBF_I, TBF_J, model,
                             Nx=64, Ny=64, L=6.0,
                             overlap_cutoff=1e-12):
        """
        Compute PE[I,J] using tensor-product Gauss–Legendre quadrature
        over a finite window determined by the overlap of the two Gaussians.
    
        TBF_I, TBF_J : your trajectory basis functions
            must have:
                .gausslist[0] (x-dim Gaussian)
                .gausslist[1] (y-dim Gaussian)
                .masses  -> array([m_x, m_y])
                .state   -> electronic state index
        model : BerryModel2DParallelTransport
    
        Returns:
            PE_IJ : complex number (matrix element H_potential-like)
        """
    
        state_I = TBF_I.state
        state_J = TBF_J.state
        massvec = np.asarray(TBF_J.masses, dtype=float)  # m_x, m_y (or same from I)
    
        # --- Quick overlap cutoff ---
        # Use the analytic overlap to skip nearly orthogonal pairs
        # (Assumes you have this helper; otherwise skip this block.)
        S_IJ = ApplyToTraj(TBF_I, TBF_J, OverlapR, False)
        if abs(S_IJ) < overlap_cutoff:
            return 0.0 + 0.0j
    
        # --- Build 1D quadrature grids in x and y ---
        gIx = TBF_I.gausslist[0]
        gJx = TBF_J.gausslist[0]
        _, x_min, x_max = self.overlap_window_1d(gIx, gJx, L=L)
        x_nodes, wx = make_legendre_grid_1d(x_min, x_max, Nx)
    
        gIy = TBF_I.gausslist[1]
        gJy = TBF_J.gausslist[1]
        _, y_min, y_max = self.overlap_window_1d(gIy, gJy, L=L)
        y_nodes, wy = make_legendre_grid_1d(y_min, y_max, Ny)
    
        # --- Precompute nuclear factors along x and y ---
        phiIcx_x, phiJ_x, dphiJdx_x = self.precompute_gaussian_1d(gIx, gJx, x_nodes)
        phiIcx_y, phiJ_y, dphiJdy_y = self.precompute_gaussian_1d(gIy, gJy, y_nodes)
    
        # For convenience, build their "overlap factors"
        # O_x[jx] = ψ_I^*(x_jx) ψ_J(x_jx)
        # O_y[jy] = ψ_I^*(y_jy) ψ_J(y_jy)
        O_x = phiIcx_x * phiJ_x
        O_y = phiIcx_y * phiJ_y
    
        # Derivative overlaps for momentum coupling:
        # Dx_x[jx] = ψ_I^*(x_jx) (∂x ψ_J)(x_jx)
        # Dy_y[jy] = ψ_I^*(y_jy) (∂y ψ_J)(y_jy)
        Dx_x = phiIcx_x * -1j*dphiJdx_x
        Dy_y = phiIcx_y * -1j*dphiJdy_y
    
        # --- Now assemble integrals ---
    
        # We'll accumulate:
        #   PE_scalar = ∫ dR χ_I^*(R) χ_J(R) [V + D + div_d]
        #   PE_pmom   = -i ∑_α (1/m_α) ∫ dR χ_I^*(R) d_α(R) ∂_α χ_J(R)
    
        PE_scalar = 0.0 + 0.0j
        PE_pmom   = 0.0 + 0.0j
    
        mx = massvec[0]
        my = massvec[1]
    
        # double loop over tensor product grid
        for ix, x in enumerate(x_nodes):
            for iy, y in enumerate(y_nodes):
                w2 = wx[ix] * wy[iy]
    
                # product of nuclear amplitudes at (x,y)
                psi_prod_xy = O_x[ix] * O_y[iy]
    
                rvec_xy = np.array([x, y], dtype=float)
    
                # V + D + div_d (matrix element for bra=state_I, ket=state_J)
                Vterm  = 0.0
                if state_I == state_J:
                    Vterm = model.V_adiabatic(rvec_xy, state_I)
    
                Dterm   = model.D(rvec_xy, massvec, state_I, state_J)
                Divterm = model.div_d(rvec_xy, massvec, state_I, state_J)
    
                scalar_here = Vterm + Dterm + Divterm
                PE_scalar  += w2 * psi_prod_xy * scalar_here
    
                # momentum coupling:
                #   -i * [ d_x / m_x * <ψ_I|∂_x ψ_J>_x * <ψ_I|ψ_J>_y
                #        + d_y / m_y * <ψ_I|ψ_J>_x   * <ψ_I|∂_y ψ_J>_y ]
                #
                # Trick: these factor in x vs y. BUT d_x,d_y depend on (x,y),
                # so we still evaluate d at this (x,y) point.
    
                d_vec = model.d(rvec_xy, state_I, state_J)  # 2-vector complex
    
                # We'll construct the needed 1D overlaps at this grid point.
                # For the current ix,iy:
                #
                # <ψ_I|∂_x ψ_J> * <ψ_I|ψ_J>_y   ~ Dx_x[ix] * O_y[iy]
                # <ψ_I|ψ_J>_x   * <ψ_I|∂_y ψ_J> ~ O_x[ix]  * Dy_y[iy]
                #
                # so the local "integrand" for the α=x piece is
                #   d_x(r) / m_x * Dx_x[ix] * O_y[iy]
                # and for α=y piece:
                #   d_y(r) / m_y * O_x[ix]  * Dy_y[iy]
                #
                # We multiply each by (-1j) and add them.
                #
                # Then don't forget weight w2.
    
                term_x = (d_vec[0] / mx) * Dx_x[ix] * O_y[iy]
                term_y = (d_vec[1] / my) * O_x[ix]  * Dy_y[iy]
    
                PE_pmom += w2 * (-1j) * (term_x + term_y)
    
        # Total PE matrix element:
        print(f"PE_scalar: {PE_scalar}")
        print(f"PE_pmom: {PE_pmom}")
        PE_IJ = PE_scalar + PE_pmom
    
        return PE_IJ

    def compute_PE_pair_fast_eps(self, TBF_I_phys, TBF_J_phys, model,
                                 eps,
                                 Nx=64, Ny=64, L=6.0,
                                 overlap_cutoff=1e-12):
        """
        Compute PE[I,J] for the eps-scaled (squeezed) packets using
        tensor-product Gauss–Legendre quadrature over truncated domains.

        This mirrors compute_PE_pair_fast, but:
          - nuclear Gaussians are replaced by their eps-squeezed versions,
          - the momentum-coupling term uses the same eps-scaling logic
            as Momentum_eps (so it stays physical instead of blowing up).

        Inputs
        ------
        TBF_I_phys, TBF_J_phys : Trajectory
            Physical trajectories (unsqueezed). We'll internally build
            squeezed copies Chi_I^eps and Chi_J^eps via squeeze_traj_for_eps.
        model : BerryModel2DParallelTransport
        eps   : float
            semiclassical squeeze parameter
        Nx,Ny : int
            number of Gauss–Legendre nodes in x and y
        L     : float
            half-width multiplier for the truncated integration window
        overlap_cutoff : float
            skip pairs with tiny overlap

        Returns
        -------
        PE_IJ : complex
            matrix element <Chi_I^eps| H_PE |Chi_J^eps>
            where H_PE includes:
                V_adiabatic (on-diagonal surfaces only),
                D (DBOC-like geometric scalar),
                div_d (∇·d/m),
                and the -i d·(p/m) coupling, with eps scaling.
        """

        # Build squeezed packets (these carry a_eff = a/eps, p_eff = p/eps)
        TBF_I_sqz = self.squeeze_traj_for_eps(TBF_I_phys, eps)
        TBF_J_sqz = self.squeeze_traj_for_eps(TBF_J_phys, eps)

        state_I = TBF_I_phys.state
        state_J = TBF_J_phys.state

        # We still want physical masses, not scaled ones
        massvec = np.asarray(TBF_J_phys.get_masses(), dtype=float)  # [m_x, m_y]

        # Quick overlap cutoff using the squeezed overlap
        S_IJ_sqz = ApplyToTraj(TBF_I_sqz, TBF_J_sqz, OverlapR, False)
        if abs(S_IJ_sqz) < overlap_cutoff:
            return 0.0 + 0.0j

        # --- Build 1D quadrature grids in x and y using the squeezed packets ---
        gIx_sqz = TBF_I_sqz.gausslist[0]
        gJx_sqz = TBF_J_sqz.gausslist[0]
        _, x_min, x_max = self.overlap_window_1d(gIx_sqz, gJx_sqz, L=L)
        x_nodes, wx = make_legendre_grid_1d(x_min, x_max, Nx)

        gIy_sqz = TBF_I_sqz.gausslist[1]
        gJy_sqz = TBF_J_sqz.gausslist[1]
        _, y_min, y_max = self.overlap_window_1d(gIy_sqz, gJy_sqz, L=L)
        y_nodes, wy = make_legendre_grid_1d(y_min, y_max, Ny)

        # Precompute χ_I^*(x) χ_J(x) etc. for squeezed packets
        # along x:
        phiIcx_x, phiJ_x, dphiJdx_x = self.precompute_gaussian_1d(
            gIx_sqz, gJx_sqz, x_nodes
        )
        # along y:
        phiIcx_y, phiJ_y, dphiJdy_y = self.precompute_gaussian_1d(
            gIy_sqz, gJy_sqz, y_nodes
        )

        # Overlap factors
        # O_x[ix] = χ_I^*(x_ix) χ_J(x_ix)
        # O_y[iy] = χ_I^*(y_iy) χ_J(y_iy)
        O_x = phiIcx_x * phiJ_x
        O_y = phiIcx_y * phiJ_y

        # Derivative overlaps for momentum coupling (squeezed)
        # Dx_x[ix] = χ_I^*(x_ix) (∂_x χ_J)(x_ix)
        # Dy_y[iy] = χ_I^*(y_iy) (∂_y χ_J)(y_iy)
        Dx_x = phiIcx_x * -1j*dphiJdx_x
        Dy_y = phiIcx_y * -1j*dphiJdy_y

        # We'll integrate:
        #   PE_scalar = ∫ dR χ_I^*(R) χ_J(R) [ V + D + div_d ]
        #   PE_pmom   = -i ∑_α (1/m_α) ∫ dR χ_I^*(R) d_α(R) (∂_α χ_J)(R)
        #
        # BUT for eps-squeezed Gaussians, we know that (∂_α χ_J^sqz)
        # has a 1/eps scale. Your Momentum_eps() compensates by
        # multiplying the matrix element by eps, so the final physical
        # "momentum" piece stays O(1).
        #
        # We'll mimic that here by multiplying Dx_x and Dy_y by eps
        # inside the integrand for the momentum term.

        PE_scalar = 0.0 + 0.0j
        PE_pmom   = 0.0 + 0.0j

        mx = massvec[0]
        my = massvec[1]

        for ix, x in enumerate(x_nodes):
            for iy, y in enumerate(y_nodes):
                w2 = wx[ix] * wy[iy]

                psi_prod_xy = O_x[ix] * O_y[iy]     # χ_I^* χ_J at (x,y)

                rvec_xy = np.array([x, y], dtype=float)

                # ----- scalar gauge pieces evaluated at (x,y) -----

                Vterm = 0.0
                if state_I == state_J:
                    Vterm = model.V_adiabatic(rvec_xy, state_I)

                Dterm   = model.D(rvec_xy, massvec, state_I, state_J)
                Divterm = model.div_d(rvec_xy, massvec, state_I, state_J)

                scalar_here = Vterm +  (Dterm + Divterm)

                PE_scalar += w2 * psi_prod_xy * scalar_here

                # ----- momentum · derivative-coupling piece -----
                # d_vec = derivative coupling vector between states,
                # evaluated at this geometry.
                d_vec = model.d(rvec_xy, state_I, state_J)  # [d_x, d_y]

                # unsqueezed integrand:
                #   (-i) * [ d_x/mx * Dx_x[ix]*O_y[iy] + d_y/my * O_x[ix]*Dy_y[iy] ]
                #
                # for eps-squeezed packets we add the same eps factor
                # that Momentum_eps() would apply to <I|p̂|J>
                # to cancel the 1/eps blowup from squeezing.
                term_x = (d_vec[0] / mx) * (eps * Dx_x[ix]) * O_y[iy]
                term_y = (d_vec[1] / my) * O_x[ix]          * (eps * Dy_y[iy])

                PE_pmom += w2 * (-1j) * (term_x + term_y)

        PE_IJ = PE_scalar + PE_pmom
        return PE_IJ
