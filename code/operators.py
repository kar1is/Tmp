# from densitymatrix import DensityMatrix
from interactions import BField, EField
from utilities import com, acom
import sympy as sp
from matrix_elements import hyperfine_magnetic_element, \
                            electric_dipole_element


# General operator class
class Operator:
    len: int
    symbols: list[object]   # Symbols used in the representaion
    mtx: list[object]       # Matrix representation
    
    def _add_sym(self, symbols: object):
        if not isinstance(symbols, list):
            symbols = [symbols]
        for symbol in symbols:
            if symbol not in self.symbols:
                self.symbols.append(symbol)


class Hamiltonian(Operator):
    def __init__(self, ρ):
        self.rho = ρ
        self.states = ρ.states
        self.len = ρ.len
        self.symbols = []

        self.mtx = sp.zeros(self.len, self.len)
        self._add_bare_energy()

    def _add_bare_energy(self):
        previous = None
        energy_number = 0

        for i,state in enumerate(self.states):
            manifold = (state.n, state.L, state.F)

            if manifold != previous:
                energy_number += 1
                previous = manifold

            self.mtx[i,i] += sp.Symbol(f"E{energy_number}")

    def add_field(self, field):
        self._add_sym(field.symbols)
        for i,state_i in enumerate(self.states):
            for j,state_j in enumerate(self.states):
                
                if isinstance(field,BField):
                    element = hyperfine_magnetic_element(
                                field, state_i, state_j)

                elif isinstance(field,EField):
                    element = electric_dipole_element(field, state_i, state_j)

                else:
                    element = 0

                self.mtx[i,j] += element


class JumpOperator:
    def __init__(self, rho, ket, bra, proportion, rate):
        self.len = rho.len
        self.mtx = sp.zeros(self.len, self.len)
        self.mtx[ket, bra] = sp.sqrt(proportion * rate)
        self.symbols = [rate]


class Lindblad(Operator):
    def __init__(self, ρ, H=None):
        self.rho = ρ
        self.len = ρ.len
        self.mtx = sp.zeros(self.len, self.len)
        self.symbols = []
        
        if H is not None:
            self.add_hamiltonian(H)

    def add_hamiltonian(self, H):
        self.mtx += -sp.I * com(H.mtx, self.rho.mtx)
        self._add_sym(H.symbols)

    def add_jump(self, C):
        """ Just add a jump operator. """
        self._add_sym(C.symbols)
        self.mtx += ( C.mtx * self.rho.mtx * C.mtx.H - \
                     sp.Rational(1, 2) * acom(C.mtx.H * C.mtx, self.rho.mtx))

    def add_term(self, term):
        """ Add arbitrary superoperator term. """
        self.mtx += term
    
    def add_transit(self, γ):
        """ Transit relaxation. """
        self._add_sym(γ)
        for state in self.rho.states:
            self.spontaneous_decay(state, state, 1, γ)
        for i in range(self.len):
            # population loss
            self.mtx[i,i] -= γ * self.rho.mtx[i,i]
    
    def spontaneous_decay(self, excited, ground, proportion, rate):
        """ Add one spontaneous transition with rate Gamma. """
        e = self.rho.states.index(excited)
        g = self.rho.states.index(ground)
        C = JumpOperator(self.rho, g, e, proportion, rate)
        self.add_jump(C)

    def get_coeff(self):
        vector = sp.Matrix([
            self.mtx[i, j]
            for i in range(self.len)
            for j in range(self.len)
        ])

        return vector.jacobian(self.rho.list)