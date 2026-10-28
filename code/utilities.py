from symbols import t
import sympy as sp
from sympy.physics.wigner import wigner_3j, wigner_6j

### Basic ###
    
def com(A: sp.MatrixBase, B: sp.MatrixBase) -> sp.MatrixBase:
    '''Matrix commutator.'''
    return A*B - B*A

def acom(A, B):
    """Matrix anticommutator."""
    return A*B + B*A

def to_list(mtx):
    """Convert the given matrix into a list in a row-major format."""
    tmp_list = []
    for i in range(mtx.shape[0]):
        for j in range(mtx.shape[1]):
            tmp_list.append(mtx[i,j])
    return sp.Matrix(tmp_list)

def branching_rate(excited, ground):

    Fe = excited.F
    Fg = ground.F
    me = excited.mF
    mg = ground.mF

    q = mg - me

    if abs(q) > 1:
        return sp.S(0)

    strength = (
        (2*Fg + 1)
        * wigner_6j(
            ground.J, excited.J, 1,
            Fe, Fg, excited.I
        )**2
        * wigner_3j(
            Fg, 1, Fe,
            -mg, q, me
        )**2
    )

    return sp.simplify(strength)


def split_frequencies(operator, omega):
    """
    Split a matrix into Fourier components.

    Returns
    -------
    dict
        n -> matrix containing the coefficient of exp(i*n*omega*t).
    """
    rows, cols = operator.shape
    freq_split_op = {}

    target = sp.I * omega * t

    def add_frequency(n):
        if n not in freq_split_op:
            freq_split_op[n] = sp.zeros(rows, cols)

    for i in range(rows):
        for j in range(cols):
            expr = sp.expand(operator[i, j].rewrite(sp.exp))

            for term in sp.Add.make_args(expr):
                n = 0
                exponential = 1

                for e in term.atoms(sp.exp):
                    exponent = sp.expand(e.args[0])
                    n_term = exponent.coeff(target)

                    if n_term != 0:
                        if not n_term.is_integer:
                            raise ValueError(
                                f"Non-integer Floquet harmonic: {n_term}"
                            )

                        n += int(n_term)
                        exponential *= e

                add_frequency(n)

                coefficient = sp.simplify(term / exponential)
                freq_split_op[n][i, j] += coefficient

    return dict(sorted(freq_split_op.items()))