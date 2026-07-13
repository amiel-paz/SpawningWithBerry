import os, math
import numpy as np
import matplotlib.pyplot as plt
from math import erf, pi, sqrt, exp
import cmath
from scipy.special import erf
from gaussfuncs import *

# Implement the routines for the Berry's phase model
class BerryModel2DParallelTransport(object):

    def __init__(self, E, B, W):
        self.E = E    # adiabatic energy scale / splitting
        self.B = B    # switching sharpness for theta(x)
        self.W = W    # phi(y) gradient / "momentum" in y
        self.ndim = 2

    def classical_energy(self, rvec, pvec, mass, state):
        inv_m = 1.0 / np.asarray(mass, dtype=float)
        G = np.diag(inv_m)
    
        # 1) bare ½ p^T G p
        T_cl = 0.5 * float(np.real(pvec.T @ (G @ pvec)))
    
        # 2) cross term: - A·(G p)  (since d_nn = i A ⇒ -i d_nn = +A)
        A = self.A(rvec, state)
        cross = float(np.real(np.dot(A * inv_m, pvec)))
    
        # 3) + ½ A·G A  (diagonal part of D)
        diagA2 = self.D_diagA2(rvec, mass, state)
    
        # 4) scalars that truly belong in Φ: BO + off-diag D + div d
        Vbo   = self.V_adiabatic(rvec, state)
        Doff  = self.D_off(rvec, mass, state)
        Divd  = self.div_d(rvec, mass, state, state)  # your model: 0 on the diagonal
    
        Phi = Vbo + Doff + Divd
    
        return T_cl + cross + diagA2 + Phi


    def H_diabatic(self, rvec):
        assert len(rvec) == 2, "Must be a vector of length 2 (x,y)" 

        x = rvec[0]
        y = rvec[1]
        theta = self.theta_of_x(x)
        phi   = self.phi_of_y(y)

        H = np.zeros((2,2), dtype=np.complex128)
        H[0,0] = cmath.cos(theta)
        H[1,1] = -H[0,0]
        H[0,1] = cmath.sin(theta) * cmath.exp(1j * phi)
        H[1,0] = H[0,1].conjugate()
        return H

    def U_adiabatic(self, rvec):
        """
        Return 2x2 unitary U(R) whose columns are adiabatic eigenvectors
        in a gauge consistent with:
            A_y(ground)  =  W * cos^2(theta/2)
            A_y(excited) =  W * sin^2(theta/2)
        State ordering: col0 = ground (state=0: energy -E), col1 = excited (state=1: +E).
        """
        x, y = float(rvec[0]), float(rvec[1])
        th = self.theta_of_x(x)
        ph = self.phi_of_y(y)
        c = math.cos(0.5*th)
        s = math.sin(0.5*th)
    
        # ground column (add +φ phase so A_y shifts by +W → W cos^2(θ/2))
        col0 = np.array([ c*np.exp(1j*ph),  -s ], dtype=np.complex128)
        # excited column (standard choice gives A_y = W sin^2(θ/2))
        col1 = np.array([ s*np.exp(1j*ph),  c ], dtype=np.complex128)
    
        # Already normalized and orthogonal, but keep the habits clean:
        U = np.column_stack([col0, col1])
        return U


    def V_adiabatic(self, rvec, state):
        assert 0 <= state < 2, "State index must be 0 or 1"

        prefac = (state - 0.5) * 2.0
        return self.E * prefac

    # Derivative coupling vector
    def d(self, rvec, bra_state, ket_state):
        assert len(rvec) == 2, "Must be a vector of length 2 (x,y)"
        assert 0 <= bra_state < 2, "State index must be 0 or 1"
        assert 0 <= ket_state < 2, "State index must be 0 or 1"

        if (bra_state == ket_state):
            # d_nn = <n|∇|n> = i A_n
            # so that p - 1j*d = p - A
            return 1j * self.A(rvec, bra_state)
        else:
            return self.NAC(rvec, bra_state, ket_state)

    # Return -1/2m times sum_I d_{bra_state,I} d_{I,ket_state}
    def D(self, rvec, mass, bra_state, ket_state):
        assert len(rvec) == 2, "Must be a vector of length 2 (x,y)" 
        assert 0 <= bra_state < 2, "State index must be 0 or 1" 
        assert 0 <= ket_state < 2, "State index must be 0 or 1" 

        D = 0
        for idx in range(2):
            D += np.dot(self.d(rvec, bra_state, idx) * (1/mass), self.d(rvec, idx, ket_state))
        return -0.5 * D

    # Return -1/2m times divergence of d
    def div_d(self, rvec, mass, bra_state, ket_state):
        """
        Compute ∇·( G d_{bra,ket} ), where
        G = diag(1/m_x, 1/m_y)
        d_{bra,ket} is the derivative coupling vector <bra|∇|ket> in Subotnik gauge.

        rvec: [x,y]
        bra_state, ket_state: 0 or 1
        mass: [m_x, m_y]  (so we can build the inverse mass metric)
        """

        x = rvec[0]
        # y = rvec[1]  # we actually don't need y numerically in this model
        mx = mass[0]
        my = mass[1]

        inv_mx = 1.0 / mx
        inv_my = 1.0 / my

        th        = self.theta_of_x(x)
        th_prime  = self.dtheta_dx_of_x(x)      # θ'(x)
        th_second = self.d2theta_dx2_of_x(x)    # θ''(x)

        # We'll build d_x and d_y analytically for each (bra_state, ket_state)
        # in Subotnik's gauge.

        if bra_state == 0 and ket_state == 0:
            # d_00 = (0, i W cos^2(θ/2))
            d_x = 0.0
            d_y = 1j * self.W * (np.cos(0.5 * th) ** 2)

        elif bra_state == 1 and ket_state == 1:
            # d_11 = (0, i W sin^2(θ/2))
            d_x = 0.0
            d_y = 1j * self.W * (np.sin(0.5 * th) ** 2)

        elif bra_state == 0 and ket_state == 1:
            # d_01 = ( θ'(x)/2 ,  (i/2) sinθ * W )
            d_x = 0.5 * th_prime
            d_y = 0.5j * np.sin(th) * self.W

        elif bra_state == 1 and ket_state == 0:
            # d_10 = - d_01*  (this gives: (-θ'/2,  + (i/2) sinθ * W) )
            d_x = -0.5 * th_prime
            d_y = 0.5j * np.sin(th) * self.W

        else:
            raise ValueError("State index must be 0 or 1")

        # Now compute divergence of (G d):
        # div = ∂_x [ inv_mx * d_x ] + ∂_y [ inv_my * d_y ].
        #
        # In this model:
        #   d_x depends only on x,
        #   d_y depends only on x,
        #   so ∂_y of (*anything*) = 0.
        #
        # So:
        #   div = (∂_x inv_mx)*d_x + inv_mx*(∂_x d_x)
        # But inv_mx is constant mass, so (∂_x inv_mx) = 0.
        #
        # So we only need inv_mx * ∂_x d_x.

        # derivative of d_x w.r.t x:
        if bra_state == 0 and ket_state == 1:
            ddx_d_x = 0.5 * th_second    # derivative of (0.5 θ')
        elif bra_state == 1 and ket_state == 0:
            ddx_d_x = -0.5 * th_second   # derivative of (-0.5 θ')
        else:
            # in the diagonal cases, d_x = 0 → ∂_x d_x = 0
            ddx_d_x = 0.0

        term_x = inv_mx * ddx_d_x

        # term_y = ∂_y [ inv_my * d_y ] = 0 because d_y = d_y(x) only.
        term_y = 0.0

        div_val = term_x + term_y  # complex scalar in general

        return -0.5 * div_val

    # ===== Geometry / helper functions =====

    # theta(x) controls mixing angle in diabatic -> adiabatic rotation
    def theta_of_x(self, x):
        # θ(x) = 0.5 π [erf(B x) + 1]
        return 0.5 * pi * (erf(self.B * x) + 1.0)

    def dtheta_dx_of_x(self, x):
        # d/dx erf(Bx) = 2/√π * B * exp(-(Bx)^2)
        # so dθ/dx = 0.5 π * (2/√π) B exp(- (Bx)^2)
        #          = √π * B * exp(- (Bx)^2)
        return sqrt(pi) * self.B * np.exp(-(self.B * x) ** 2)

    def phi_of_y(self, y):
        # φ(y) = W * y
        return self.W * y

    # Berry connection A_n(R) for adiabatic state n.
    # In this gauge, only A_y is nonzero.
    def Ay_of_x(self, x, state):
        th = self.theta_of_x(x)

        if state == 0:
          # ground: A_y = W * cos^2(θ/2)
          return self.W * (np.cos(0.5 * th) ** 2)

        elif state == 1:
          # excited: A_y = W * sin^2(θ/2)
          return self.W * (np.sin(0.5 * th) ** 2)

        else:
          raise ValueError("State index must be 0 or 1")


    def dAy_dx_of_x(self, x, state):
        th       = self.theta_of_x(x)
        th_prime = self.dtheta_dx_of_x(x)

        # d/dx cos^2(th/2) = -(1/2) sin(th) * th'(x)
        # d/dx sin^2(th/2) = +(1/2) sin(th) * th'(x)
        #
        # ORIGINAL:
        #   state 0 → -(1/2) sin(th) th'
        #   state 1 → +(1/2) sin(th) th'
        #
        # After flipping Ay for state 1, we ALSO flip its derivative sign.
        if state == 0:
            return (-1.0) * 0.5 * np.sin(th) * th_prime * self.W
        elif state == 1:
            return (+1.0) * 0.5 * np.sin(th) * th_prime * self.W
        else:
            raise ValueError("State index must be 0 or 1")

    # Vector potential A_n(r) for a given adiabatic state.
    # returns ndarray([A_x, A_y]) with A_x = 0, A_y = Ay(x)
    def A(self, rvec, state):
        x = rvec[0]
        return np.asarray(
            [0.0, self.Ay_of_x(x, state)],
            dtype=np.complex128
        )

    # Jacobian of A wrt coordinates: ∇A as a 2x2 matrix
    # rows = A component index, cols = spatial derivative direction
    #   J[0,0] = ∂_x A_x,  J[0,1] = ∂_y A_x
    #   J[1,0] = ∂_x A_y,  J[1,1] = ∂_y A_y
    def jacobian_A(self, rvec, state):
        x = rvec[0]

        dAx_dx = 0.0
        dAx_dy = 0.0

        dAy_dx = self.dAy_dx_of_x(x, state)
        dAy_dy = 0.0

        J = np.asarray(
            [[dAx_dx, dAx_dy],
             [dAy_dx, dAy_dy]],
            dtype=np.complex128
        )
        return J

    # Antisymmetric curvature matrix Ω(R) = (∇A)^T - (∇A)
    # In 2D, Ω = [[0,  ∂_x A_y - ∂_y A_x],
    #             [-(∂_x A_y - ∂_y A_x), 0]]
    # With A_x = 0, A_y = A_y(x), that's just dAy/dx in the off-diagonals.
    def Omega(self, rvec, state):
        J = self.jacobian_A(rvec, state)
        Omega_mat = (J.T - J)
        return Omega_mat

    def d2theta_dx2_of_x(self, x):
        # θ'(x) = sqrt(pi) * B * exp(-(B x)^2)
        # θ''(x) = d/dx [ sqrt(pi) * B * exp(-(B x)^2) ]
        #        = sqrt(pi) * B * exp(-(B x)^2) * (-2 (B x) * B)
        #        = -2 * sqrt(pi) * (B**3) * x * exp(-(B x)**2)
        return -2.0 * sqrt(pi) * (self.B ** 3) * x * np.exp(-(self.B * x) ** 2)

    def D_diagA2(self, rvec, mass, state):
        inv_m = 1.0 / np.asarray(mass, dtype=float)
        A = self.A(rvec, state)              # shape (2,)
        # 0.5 * A·G^{-1}A  (this is real)
        return 0.5 * float(np.real(np.dot(A * inv_m, A)))
    
    def D_off(self, rvec, mass, state):
        # -0.5 * sum_{I != n} d_{nI}·G^{-1} d_{In}
        inv_m = 1.0 / np.asarray(mass, dtype=float)
        n = state
        acc = 0.0 + 0.0j
        for I in range(2):
            if I == n: 
                continue
            acc += np.dot(self.d(rvec, n, I) * inv_m, self.d(rvec, I, n))
        return -0.5 * float(np.real(acc))


    # ===== Gradient of scalar potential and gauge force =====

    def gradD_total(self, rvec, mass, state):
        """
        Return +∇U_a(r) as a 2-vector, where in your old convention
        U_a(r) = D_{aa}(r).
    
        Here:
          - mass: array-like [m_x, m_y]
          - state: 0 or 1 (adiabatic surface index)
    
        This is the *full* gradient you were previously using for gradU,
        i.e. it still includes both the off-diagonal Born-Huang (geometric)
        piece and the diagonal A^2/(2m) piece.
        We'll subtract the diagonal A^2/(2m) gradient separately when we
        build the corrected gradU.
        """
    
        x  = rvec[0]
        mx = mass[0]
        my = mass[1]
    
        th       = self.theta_of_x(x)        # θ(x)
        th_prime = self.dtheta_dx_of_x(x)    # θ'(x)
        th_2p    = self.d2theta_dx2_of_x(x)  # θ''(x)
    
        # handy shorthands
        c = np.cos(0.5 * th)   # cos(θ/2)
        s = np.sin(0.5 * th)   # sin(θ/2)
    
        # pieces common to both states:
        # (th_prime * th_2p)/(4 mx)  is from the θ'^2 / (4m_x) structure,
        # (W^2/(4 my)) * sinθ cosθ * θ'  is from the sin^2(θ/2), cos^2(θ/2) pieces, etc.
        term_common = (th_prime * th_2p) / (4.0 * mx) \
                    + (self.W**2 / (4.0 * my)) * np.sin(th) * np.cos(th) * th_prime
    
        if state == 0:
            # derivative of D_00 wrt x
            # extra state-dependent piece:  -(W^2/my) * cos^3(θ/2)*sin(θ/2) * θ'
            term_state = -(self.W**2 / my) * (c**3) * s * th_prime
            dU_dx = term_state + term_common
    
        elif state == 1:
            # derivative of D_11 wrt x
            # extra state-dependent piece: +(W^2/my) * sin^3(θ/2)*cos(θ/2) * θ'
            term_state = +(self.W**2 / my) * (s**3) * c * th_prime
            dU_dx = term_state + term_common
    
        else:
            raise ValueError("State index must be 0 or 1")
    
        # U depends only on x in this gauge, so ∂U/∂y = 0
        grad_vec = np.asarray([dU_dx, 0.0], dtype=np.complex128)
        return grad_vec


    def gradU(self, rvec, mass, state):
        """
        Return +∇Φ_n(r) where Φ_n = V_BO + D_off + div d.
        (Diagonal ½ A·G A is *not* included; it lives in the kinetic.)
        """
        # 1) total grad of your previous D (reuse your analytic expression)
        gradD_total = self.gradD_total(rvec, mass, state)   # <- this is your current code
        # If you don't have it factored, gradD_total == old self.gradU(...)
    
        # 2) subtract grad of ½ A·G A  = (∇A)^T (G A)
        inv_m = 1.0 / np.asarray(mass, dtype=float)
        A     = self.A(rvec, state)
        J     = self.jacobian_A(rvec, state)                 # rows: A comp, cols: coord
        grad_diagA2 = J.T @ (inv_m * A)
    
        # 3) add ∇V_BO (zero here) and ∇(div d) if nonzero in your model
        grad_Vbo = np.zeros_like(A, dtype=np.complex128)     # flat in your gauge
        grad_divd = np.zeros_like(A, dtype=np.complex128)    # 0 on-diagonal in your model
    
        # 4) net +∇Φ
        return (gradD_total - grad_diagA2 + grad_Vbo + grad_divd)
        #return (gradD_total + grad_Vbo + grad_divd)


    def gauge_force(self, rvec, mass, state):
        """
        Return the 'electric gauge force' F_scalar(r) = -∇U_a(r)
        for the active adiabatic state.
        This is what you'd use as the conservative force term in the Boris step:
            m dv/dt = ... + F_scalar
        """
        grad_vec = self.gradU(rvec, mass, state)
        F = -grad_vec  # minus gradient
        # physically this should be real
        return F.real


    # ===== Equations of motion pieces =====

    def get_rdot(self, G, rvec, pvec, state):
        # rdot = G @ (p + A)
        A = self.A(rvec, state)
        return G @ (pvec + A)
    
    def get_pdot(self, G, rvec, pvec, state):
        J = self.jacobian_A(rvec, state)   # ∇A
        A = self.A(rvec, state)
        rdot = G @ (pvec + A)
        mass = np.reciprocal(np.diag(G))
        gradU_vec = self.gradU(rvec, mass, state)  # = +∇Φ
    
        # For H = 1/2 (p + A)^T G (p + A) + Φ:
        # ∂H/∂r = J^T rdot + ∇Φ   →  pdot = -∂H/∂r = -J^T rdot - ∇Φ
        return -(J.T @ rdot) - gradU_vec

    def get_adot(self, G, rvec, pvec, state):
        # placeholder for internal phase 'a' if you track it
        return np.zeros(len(rvec))

    def get_gammadot(self, G, rvec, pvec, state):
        # placeholder for geometric phase 'gamma' if you track it
        return np.zeros(len(rvec))

    def NAC(self, rvec, bra_state, ket_state):
        x = rvec[0]
        th       = self.theta_of_x(x)
        th_prime = self.dtheta_dx_of_x(x)  # dθ/dx

        # off-diagonal derivative coupling ⟨bra|∇|ket⟩
        # x-component:  0.5 * θ'(x)
        # y-component:  0.5 i sin(θ) W
        NAC_vec = np.asarray(
            [0.5 * th_prime,
             0.5j * np.sin(th) * self.W],
            dtype=np.complex128
        )

        if bra_state < ket_state:
            return NAC_vec
        else:
            return -(NAC_vec.conjugate())

    def eval_integrals(self, traj_1, traj_2, thresh=1.0e-8, hbar=1, KE=False):
        curr_val = 0.0 + 0.0j
        traj_1_x = traj_1.gausslist[0]
        traj_1_y = traj_1.gausslist[1]
        traj_2_x = traj_2.gausslist[0]
        traj_2_y = traj_2.gausslist[1]
        S_x = OverlapR(traj_1_x,traj_2_x,False) 
        S_y = OverlapR(traj_1_y,traj_2_y,False) 
        P_x = Momentum(traj_1_x,traj_2_x,False) 
        P_y = Momentum(traj_1_y,traj_2_y,False)
        m_x = traj_2_x.m 
        m_y = traj_2_y.m 
        #print(f"S_x: {abs(S_x)}")
        #print(f"S_y: {abs(S_y)}")
        #print(f"P_x: {abs(P_x)}")
        #print(f"P_y: {abs(P_y)}")
        traj_1_params = [traj_1_x.R, traj_1_x.a, -traj_1_x.P, -traj_1_x.gamma] # Conjugated 
        traj_2_params = [traj_2_x.R, traj_2_x.a, traj_2_x.P, traj_2_x.gamma]
        A_comb, B_comb, C_comb = self.return_quadratic([traj_1_params, traj_2_params])
        hbar2 = hbar*hbar
        prefactor_norm = traj_1_x.get_normalization() * traj_2_x.get_normalization()
        _, lb, ub = self.overlap_window_1d(traj_1_x, traj_2_x)
        if (traj_1.state == traj_2.state == 0):
            # Momentum terms (quadrature)
            #print(f"A_comb, B_comb, C_comb: {A_comb}, {B_comb}, {C_comb}")
            # (Right)
            precomputed = None
            prefactor_hbar_mass = (2) * (-1j * hbar) * (0.5/m_y)
            prefactor_d = 1j * self.W
            prefactor = prefactor_hbar_mass * prefactor_d * prefactor_norm
            if not (abs(P_y) * abs(S_x) < (thresh / max(abs(prefactor), 1e-300))):
                if precomputed is None:
                    precomputed, _ = gh_integral_adaptive (
                        lambda x: np.cos(0.5 * self.theta_of_x(x))**2,
                                  A_comb,
                                  B_comb,
                                  C_comb)
                curr_val += prefactor * P_y * precomputed
                pmom = prefactor * P_y * precomputed
                #print(f"PE_pmom: {P_y * prefactor * precomputed}")
            # Single state divergence terms are 0
            # Two-sided derivative term
            # Dx (analytic)
            D_params = [0, 2 * self.B**2, 0, 0]
            prefactor_hbar_mass = (1) * (-hbar2) * (0.5/m_x)
            prefactor = -(math.pi/4) * self.B**2 * prefactor_hbar_mass * prefactor_norm
            param_list = [traj_1_params, traj_2_params, D_params]
            curr_val += prefactor * S_y * self.generalized_gaussian(param_list)
            #print(f"Dx: {S_y * prefactor * self.generalized_gaussian(param_list)}") 
            # Dy (quadrature)
            prefactor_hbar_mass = (1) * (-hbar2) * (0.5/m_y)
            prefactor = -self.W**2 * prefactor_hbar_mass * prefactor_norm
            if not (abs(S_y) * abs(S_x) < (thresh / max(abs(prefactor), 1e-300))):
                if precomputed is None:
                    precomputed, _ = gh_integral_adaptive (
                        lambda x: np.cos(0.5 * self.theta_of_x(x))**2,
                                  A_comb,
                                  B_comb,
                                  C_comb)
                curr_val += prefactor * S_y * precomputed
            #print(f"Dy: {S_y * prefactor * precomputed}") 
            curr_val += self.V_adiabatic(0.0,0) * S_x * S_y
            #print(f"PE_scalar: {curr_val - pmom}")
            if (KE):
                curr_val += Kinetic(traj_1_x,traj_2_x,False) * S_y + S_x *  Kinetic(traj_1_y,traj_2_y,False)
        elif (traj_1.state == traj_2.state == 1):
            # Momentum terms (quadrature)
            #print(f"A_comb, B_comb, C_comb: {A_comb}, {B_comb}, {C_comb}")
            # (Right)
            precomputed = None
            prefactor_hbar_mass = (2) * (-1j * hbar) * (0.5/m_y)
            prefactor_d = 1j * self.W
            prefactor = prefactor_hbar_mass * prefactor_d * prefactor_norm
            if not (abs(P_y) * abs(S_x) < (thresh / max(abs(prefactor), 1e-300))):
                if precomputed is None:
                    precomputed, _ = gh_integral_adaptive (
                        lambda x: np.sin(0.5 * self.theta_of_x(x))**2,
                                  A_comb,
                                  B_comb,
                                  C_comb)
                curr_val += prefactor * P_y * precomputed
                pmom = prefactor * P_y * precomputed
                #print(f"PE_pmom: {P_y * prefactor * precomputed}")
            # Single state divergence terms are 0
            # Two-sided derivative term
            # Dx (analytic)
            D_params = [0, 2 * self.B**2, 0, 0]
            prefactor_hbar_mass = (1) * (-hbar2) * (0.5/m_x)
            prefactor = -(math.pi/4) * self.B**2 * prefactor_hbar_mass * prefactor_norm
            param_list = [traj_1_params, traj_2_params, D_params]
            curr_val += prefactor * S_y * self.generalized_gaussian(param_list)
            #print(f"Dx: {S_y * prefactor * self.generalized_gaussian(param_list)}") 
            # Dy (quadrature)
            prefactor_hbar_mass = (1) * (-hbar2) * (0.5/m_y)
            prefactor = -self.W**2 * prefactor_hbar_mass * prefactor_norm
            if not (abs(S_y) * abs(S_x) < (thresh / max(abs(prefactor), 1e-300))):
                if precomputed is None:
                    precomputed, _ = gh_integral_adaptive (
                        lambda x: np.sin(0.5 * self.theta_of_x(x))**2,
                                  A_comb,
                                  B_comb,
                                  C_comb)
                curr_val += prefactor * S_y * precomputed
            #print(f"Dy: {S_y * prefactor * precomputed}") 
            curr_val += self.V_adiabatic(0.0,1) * S_x * S_y
            #print(f"PE_scalar: {curr_val - pmom}")
            if (KE):
                curr_val += Kinetic(traj_1_x,traj_2_x,False) * S_y + S_x *  Kinetic(traj_1_y,traj_2_y,False)
        else:
            # Momentum terms
            precomputed = None
            # pdotd_x (analytic)
            prefactor_hbar_mass = (2) * (-hbar2) * (0.5/m_x)
            d01x_params = [0, self.B**2, 0, 0]
            prefactor1 = -2*traj_2_x.a
            prefactor2 = 2*traj_2_x.a*traj_2_x.R + 1j*traj_2_x.P
            prefactor_all = 0.5 * math.sqrt(math.pi) * self.B * prefactor_hbar_mass * prefactor_norm 
            param_list = [traj_1_params, traj_2_params, d01x_params]
            curr_val += S_y * prefactor_all * (prefactor1 * self.generalized_gaussian_times_x(param_list) \
                        + prefactor2 * self.generalized_gaussian(param_list)) 
            pmom = S_y * prefactor_all * (prefactor1 * self.generalized_gaussian_times_x(param_list) \
                        + prefactor2 * self.generalized_gaussian(param_list)) 
            # pdotd_y (quadrature)
            prefactor_hbar_mass = (2) * (-1j * hbar) * (0.5/m_y)
            prefactor = 0.5j * self.W * prefactor_hbar_mass * prefactor_norm
            if not (abs(P_y) * abs(S_x) < (thresh / max(abs(prefactor), 1e-300))):
                if precomputed is None:
                    precomputed, _ = gh_integral_adaptive (
                        lambda x: np.sin(self.theta_of_x(x)),
                                  A_comb,
                                  B_comb,
                                  C_comb, tol = 1.0e-10)
                curr_val += P_y * prefactor * precomputed
                pmom += P_y * prefactor * precomputed
            #print(f"PE_mom: {pmom}")
            # Divergence terms
            # div_dx (analytic)
            prefactor_hbar_mass = (1) * (-hbar2) * (0.5/m_x)
            prefactor = -self.B**3 * math.sqrt(math.pi) * prefactor_hbar_mass * prefactor_norm
            curr_val += S_y * prefactor * self.generalized_gaussian_times_x(param_list)
            # Two-sided derivative term
            # Dy (quadrature)
            prefactor_hbar_mass = (1) * (-hbar2) * (0.5/m_y)
            prefactor = -0.5 * self.W**2 * prefactor_hbar_mass * prefactor_norm 
            if not (abs(S_y) * abs(S_x) < (thresh / max(abs(prefactor), 1e-300))):
                if precomputed is None:
                    precomputed, _ = gh_integral_adaptive (
                        lambda x: np.sin(self.theta_of_x(x)),
                                  A_comb,
                                  B_comb,
                                  C_comb, tol = 1.0e-10)
                curr_val += S_y * prefactor * precomputed
            #print(f"PE_scalar: {curr_val - pmom}")
            if (traj_1.state > traj_2.state):
                curr_val = curr_val.conjugate()
        return curr_val
    
    def return_quadratic(self, param_list):
        """
        Each entry is (mu_k, a_k, b_k, c_k) representing
        exp( -a_k (x - mu_k)**2 + i b_k (x - mu_k) + c_k ).
        """
        A = 0.0                # real
        B = 0.0 + 0.0j         # complex
        C = 0.0 + 0.0j         # complex
        for params in param_list:
            assert len(params) == 4, (
                "Should have four parameters for each Gaussian, corresponding to "
                "(mu_k, a_k, b_k, c_k): exp(-a_k(x - mu_k)^2 + i b_k (x - mu_k) + c_k)"
            )
            mu_k, a_k, b_k, c_k = params
            # a_k must be real and > 0
            assert not isinstance(a_k, complex) or a_k.imag == 0.0, "a_k should be real"
            assert float(a_k) > 0.0, "a_k should be positive"
    
            # expand:
            # -a (x - mu)^2 + i b (x - mu) + c
            # = -a x^2 + (2 a mu + i b) x + (-a mu^2 - i b mu + c)
            A += a_k
            B += 2.0 * a_k * mu_k + 1j * b_k
            C += -a_k * (mu_k ** 2) - 1j * b_k * mu_k + c_k
    
        return A, B, C
    
    def generalized_gaussian(self, param_list):
        A, B, C = self.return_quadratic(param_list)
        # integral of exp(-A x^2 + B x + C) from -inf to inf
        return math.sqrt(math.pi / A) * cmath.exp(C + (B * B) / (4.0 * A))
    
    def generalized_gaussian_times_x(self, param_list):
        A, B, C = self.return_quadratic(param_list)
        I0 = self.generalized_gaussian(param_list)
        # ∫ x e^{-A x^2 + Bx + C} dx = (B / (2A)) * ∫ e^{-A x^2 + Bx + C} dx
        return (B / (2.0 * A)) * I0

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
