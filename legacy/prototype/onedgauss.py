import copy
import sys
import math
import os
import re
import numpy as np

class OneD_Gauss(object):

    def __init__(self, R, P, a, m, gamma, coeff, **kwargs):
        self.R = R # initial position
        self.P = P # initial momentum
        self.a = a # fixed invwidth
        self.m = m # fixed mass
        self.gamma = gamma # initial phase
        self.coeff = coeff # initial coefficient

    def update_a(self, a):
        self.a = a

    def update_m(self, m):
        self.m = m

    def update_R(self, R):
        self.R = R

    def update_P(self, P):
        self.P = P

    def update_gamma(self, gamma):
        self.gamma = gamma

    def update_coeff(self, coeff):
        self.coeff = coeff

    def get_values(self, x):
        R = self.R
        P = self.P
        a = self.a
        gamma = self.gamma
        coeff = self.coeff
        return coeff * math.sqrt(math.sqrt(2*a/math.pi)) * np.exp(-a*(x-R)*(x-R) + 1j*P*(x-R) + 1j*gamma)

    def get_normalization(self):
        a = self.a
        return math.sqrt(math.sqrt(2*a/math.pi))

    def get_dvalues_dx(self, x):
        R = self.R
        P = self.P
        a = self.a
        return (-2*a*(x-R) + 1j*P) * self.get_values(x)

    def get_d2values_dx2(self, x):
        R = self.R
        P = self.P
        a = self.a
        pref1 = -2.0 * a
        pref2 = (-2.0 * a * (x - R) + 1j * P)**2
        return (pref1 + pref2) * self.get_values(x)

    def get_values_conjugate(self, x):
        R = self.R
        P = self.P
        a = self.a
        gamma = self.gamma
        coeff = self.coeff
        return coeff.conjugate() * math.sqrt(math.sqrt(2*a/math.pi)) * np.exp(-a*(x-R)*(x-R) - 1j*P*(x-R) - 1j*gamma)

    def get_ft_values(self, p):
        R = self.R
        P = self.P
        a = self.a
        gamma = self.gamma
        coeff = self.coeff
        return coeff * math.sqrt(math.sqrt(1/(2*a*math.pi))) * np.exp(-(p-P)*(p-P)/(4*a) - 1j*p*R + 1j*gamma)

    def get_ft_values_conjugate(self, p):
        R = self.R
        P = self.P
        a = self.a
        gamma = self.gamma
        coeff = self.coeff
        return coeff.conjugate() * math.sqrt(math.sqrt(1/(2*a*math.pi))) * np.exp(-(p-P)*(p-P)/(4*a) + 1j*p*R - 1j*gamma) 

    def print_properties(self):
        R = self.R
        P = self.P
        a = self.a
        gamma = self.gamma
        coeff = self.coeff
        print("--START properties")
        print("Current position: {:16.10f}".format(R))
        print("Current momentum: {:16.10f}".format(P))
        print("Current invwidth: {:16.10f}".format(a))
        print("Current phase: {:16.10f}".format(gamma))
        print("Current coefficient: ({:16.10f},{:16.10f})".format(coeff.real,coeff.imag))
        print("--END properties")

    def verlet_step(self, V, dVdx, tstep, args_dict):
        R = self.R
        P = self.P
        a = self.a
        m = self.m
        gamma = self.gamma

        # Momentum half step
        new_P_0p5 = P + 0.5 * tstep * -dVdx(R,args_dict)
        self.update_P(new_P_0p5)
        P = self.P

        # Position full step
        new_R = R + tstep * (P/m)
        self.update_R(new_R)
        R = self.R

        # Momentum half step
        new_P_0p5 = P + 0.5 * tstep * -dVdx(R,args_dict)
        self.update_P(new_P_0p5)
        P = self.P

        # Gamma update
        gamma_deriv = 0.5*P*P - V(R,args_dict)
        new_gamma = gamma + tstep * gamma_deriv
        self.update_gamma(new_gamma)
        return
