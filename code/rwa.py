import sympy as sp
from copy import deepcopy
from symbols import t

def RWA(operator, rotation, omega):
    """
    Apply rotating-frame transformation and remove fast oscillating terms.

    H' = U† H U - i U† dU/dt
    """

    op = deepcopy(operator)

    op.mtx = (
        rotation.H * op.mtx * rotation
        - sp.I * rotation.H * sp.diff(rotation, t)
    )

    for i in range(op.len):
        for j in range(op.len):
            op.mtx[i, j] = keep_static_terms(op.mtx[i, j], omega)

    return op


def keep_static_terms(expr, omega):
    """
    Keep only terms with zero harmonic frequency.
    """

    expr = sp.expand(expr.rewrite(sp.exp))

    result = 0
    target = sp.I * omega * t

    for term in sp.Add.make_args(expr):

        harmonic = 0

        for exponential in term.atoms(sp.exp):
            exponent = sp.expand(exponential.args[0])

            # coefficient of i*omega*t
            harmonic += exponent.coeff(target)

        if harmonic == 0:
            result += term

    return sp.simplify(result)

def optical_rotor(rho, omega_R):
    """
    Rotates excited states by exp(-i omega_R t).
    """

    U = sp.eye(rho.len)

    for i, state in enumerate(rho.states):

        if state.L == 1:     # replace later with explicit excited manifold flag
            U[i, i] = sp.exp(-sp.I * omega_R * t)

    return U