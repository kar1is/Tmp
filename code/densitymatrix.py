import sympy as sp
from atom import MagneticSublevel, HyperfineState


class DensityMatrix:

    def __init__(self, ground, excited):
        self.ground_spec = ground
        self.excited_spec = excited

        # Expand manifolds into magnetic sublevels
        self.ground_states = self._expand(ground)
        self.excited_states = self._expand(excited)
        self.states = self.ground_states + self.excited_states

        self.len = len(self.states)

        self._create_mtx()
        self._to_list()

    def _expand(self, state):
        """Expand given manifold into its states."""
        if isinstance(state, HyperfineState):
            return state.sublevels()
        if isinstance(state, MagneticSublevel):
            return [state]
        if isinstance(state, (list, tuple)):
            states = []
            for item in state:
                states.extend(self._expand(item))
            return states

        raise TypeError(f"Cannot construct density-matrix \
                        basis from {type(state)}")

    def _create_mtx(self):
        self.mtx = sp.zeros(self.len, self.len)
    
        for i in range(self.len):
            for j in range(self.len):
                self.mtx[i, j] = sp.Symbol(
                    f"ρ_{i+1}_{j+1}",
                    commutative=True
                )

    def _to_list(self):

        self.list = sp.Matrix([
            self.mtx[i, j]
            for i in range(self.len)
            for j in range(self.len)
        ])

    def __repr__(self):
        return sp.sstr(self.mtx)

    def _sympy_(self):
        return self.mtx