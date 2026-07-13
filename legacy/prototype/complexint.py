import math
import numpy as np
import scipy
from scipy import integrate
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss  # Gauss–Legendre nodes/weights
from numpy.polynomial.hermite import hermval, hermgauss

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        #return scipy.real(func(x))
        return (func(x)).real
    def imag_func(x):
        #return scipy.imag(func(x))
        return (func(x)).imag
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def integrate_nd(func, ndim):
    """
    Numerically integrate func(R) over all nuclear coordinates R,
    treating each coordinate as running from -inf to +inf.

    func : callable(Rvec) -> complex
        Rvec is a length-ndim numpy array of floats
    ndim : int
        number of nuclear DOFs

    returns complex integral value
    """
    def recurse(dim, prefix_coords):
        if dim == ndim:
            # we've chosen all coordinates; evaluate integrand
            R = np.asarray(prefix_coords, dtype=float)
            return func(R)

        def integrand_1d(x_d):
            return recurse(dim+1, prefix_coords + [x_d])

        val, _, _ = complex_quadrature(
            integrand_1d,
            -np.inf, np.inf,
            limit=200  # can tune
        )
        return val

    return recurse(0, [])

def make_legendre_grid_1d(x_min, x_max, N):
    """
    Produce Gauss–Legendre nodes/weights adapted to [x_min, x_max].

    Returns:
        x_nodes: (N,) array
        w_nodes: (N,) array
    """
    # nodes on [-1,1], weights on [-1,1]
    xi, wi = leggauss(N)

    # affine map t in [-1,1] -> x in [x_min, x_max]
    xm = 0.5*(x_max + x_min)
    xr = 0.5*(x_max - x_min)

    x_nodes = xm + xr*xi          # (N,)
    w_nodes = xr * wi             # (N,)  (Jacobian factor)
    return x_nodes, w_nodes

def gh_integral_fixed_nodes(f, A, B, C, n_nodes):
    """
    Compute
        I = ∫_{-∞}^{∞} f(x) * exp(-A x^2 + B x + C) dx
    with:
        A : real, A > 0
        B : complex ok
        C : complex ok
    f : callable that accepts a *real* numpy array x and returns array-compatible values.

    This matches the structure your real-line quad was doing:
    - we never evaluate f at complex x
    - we only let the exp(...) carry the complex bits
    - we let GH provide the exp(-y^2) weight
    """

    A_real = float(np.real(A))
    if A_real <= 0.0 or abs(np.imag(A)) > 1e-14:
        raise ValueError("A must be real and > 0 for this GH routine.")

    # standard GH nodes/weights: ∫ e^{-y^2} g(y) dy ≈ Σ w_i g(y_i)
    y, w = hermgauss(int(n_nodes))

    # map y → x
    # x = y / sqrt(A)
    sqrtA = math.sqrt(A_real)
    x = y / sqrtA

    # build the non-Gaussian part on *real* x
    fx = f(x)

    # the remaining exponential piece:
    # exp(-A x^2 + B x + C) = exp( (B/√A) y + C )
    # because -A x^2 = -A (y/√A)^2 = -y^2, which GH already handles
    expo = np.exp((B / sqrtA) * y + C)

    # integral = (1/√A) * Σ w_i [ f(x_i) * exp((B/√A) y_i + C) ]
    I = (1.0 / sqrtA) * np.dot(w, fx * expo)

    return I


def gh_integral_adaptive(f, A, B, C, tol=1e-8, max_nodes=64, start_nodes=16):
    """
    Adaptive wrapper around gh_integral_fixed_nodes.
    Doubles nodes: 16 → 32 → 64 (by default) until relative change < tol.
    """
    if start_nodes > max_nodes:
        start_nodes = max_nodes

    n = start_nodes
    I_prev = gh_integral_fixed_nodes(f, A, B, C, n)

    while True:
        n_next = min(2 * n, max_nodes) if n < max_nodes else max_nodes
        if n_next == n:
            # can't grow further
            return I_prev, n

        I_next = gh_integral_fixed_nodes(f, A, B, C, n_next)

        diff = abs(I_next - I_prev)
        denom = max(abs(I_next), abs(I_prev), 1.0)

        if diff <= tol * denom:
            return I_next, n_next

        n = n_next
        I_prev = I_next
