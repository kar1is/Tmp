import sympy as sp
from sympy.physics.wigner import wigner_3j, wigner_6j


def hyperfine_self_interaction(state1, state2):
    if state1 != state2:
        return sp.S(0)
    
    A = 2 * sp.pi * state1.A
    B = 2 * sp.pi * state1.B
    C = 2 * sp.pi * state1.C
    
    return 
    
def hyperfine_magnetic_element(field, state1, state2):
    mu_B = 2 * sp.pi * sp.Float(state1.atom.constants['mu_B_h_MHz_G'])

    if (state1.n != state2.n
        or state1.F != state2.F
        or state1.L != state2.L
        or state1.J != state2.J):
            return sp.S(0)

    F = state1.F
    m1 = state1.mF
    m2 = state2.mF
    gF = state1.gF

    q = m1 - m2

    if q == +1:
        mu_q = (-gF * mu_B / sp.sqrt(2) * sp.sqrt(F*(F+1) - m2*(m2+1)))
        field_q = (field.x - sp.I * field.y) / sp.sqrt(2)
    elif q == 0:
        mu_q = gF * mu_B * m2
        field_q = field.z
    elif q == -1:
        mu_q = (+gF * mu_B / sp.sqrt(2) * sp.sqrt(F*(F+1) - m2*(m2-1)))
        field_q = -(field.x + sp.I * field.y) / sp.sqrt(2)
    else:
        return sp.S(0)

    return sp.simplify(-(-1)**q * mu_q * field_q)


def electric_dipole_element(field, state1, state2):

    if abs(state1.L - state2.L) != 1:
        return sp.S(0)
    
    m1 = state1.mF
    m2 = state2.mF
    
    q = m1 - m2
    
    if q == -1:
        field_q = -(field.x + sp.I * field.y) / sp.sqrt(2)
    elif q == 0:
        field_q = field.z
    elif q == 1:
        field_q = (field.x - sp.I * field.y) / sp.sqrt(2)
    else:
        return sp.S(0)

    r_q = electric_mtx_element(state1, state2, q)
    return sp.simplify(-sp.Pow(-1, q) * r_q * field_q)


def electric_mtx_element(state1, state2, q):
    """ Uses Steck transition dipole matrix element. """
    element = None
    transition = find_transition(state1, state2)
    reduced_element_J = transition['dipole_matrix_element_au']
    reduced_element_F = reduced_element_J * \
        (-1)**(state2.F + state1.J + 1 + state1.I) * \
        sp.sqrt((2*state2.F+1) * (2*state1.J+1)) * \
        wigner_6j(state1.J, state2.J, 1, state2.F, state1.F, state2.I)
    element = reduced_element_F * \
        (-1)**(state2.F - 1 + state1.mF) * \
        sp.sqrt(2*state1.F+1) * \
        wigner_3j(state2.F, 1, state1.F, state2.mF, q, -state1.mF)
    return element


def find_transition(state1, state2):
    """ Find D1/D2 transition in atom database. """
    transitions = state1.atom.transitions

    for name, data in transitions.items():
        lower = data['lower']
        upper = data['upper']

        if {state1.level.name, state2.level.name} == {lower, upper}:
            return data

    raise ValueError('No transition found')