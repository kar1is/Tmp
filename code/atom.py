# atom.py

import json
from fractions import Fraction
from dataclasses import dataclass
import sympy as sp


class Atom:
    def __init__(self, filename):
        with open(filename, "r") as f:
            self.data = json.load(f)

    def atomic_level(self, name):
        data = self.level(name)
    
        return AtomicLevel(
            atom=self,
            name=name,
            n=data['n'],
            L=data['L'],
            J=sp.Rational(str(data['J'])),
            S=sp.Rational(str(data['S'])),
            I=sp.Rational(str(self.nuclear_spin)),
            hyperfine=data['hyperfine'],
            F_data=data['F'],
        )

    @property
    def name(self):
        return self.data["atom"]["name"]

    @property
    def isotope(self):
        return self.data["atom"]["isotope"]

    @property
    def nuclear_spin(self):
        return Fraction(str(self.data["atom"]["nuclear_spin"]))

    @property
    def constants(self):
        return self.data["constants"]

    @property
    def levels(self):
        return self.data["levels"]

    @property
    def transitions(self):
        return self.data["transitions"]

    def level(self, name):
        return self.levels[name]

    def transition(self, name):
        return self.transitions[name]


class AtomParameters:
    def __init__(self, atom):
        self.atom = atom

        self.mu_B_h_MHz_G = sp.Float(
            atom.constants["mu_B_h_MHz_G"]
        )
        self.g_L = sp.Float(atom.constants["g_L"])
        self.g_S = sp.Float(atom.constants["g_S"])
        self.g_I = sp.Float(atom.constants["g_I"])

    def level(self, name):
        return self.atom.level(name)

    def transition(self, name):
        return self.atom.transition(name)


@dataclass(frozen=True)
class AtomicLevel:
    atom: Atom
    name: str
    n: int
    L: int
    J: sp.Expr
    S: sp.Expr
    I: sp.Expr
    hyperfine: dict
    F_data: dict

    @property
    def gJ(self):
        return sp.simplify(
            1 + (
                self.J * (self.J + 1)
                + self.S * (self.S + 1)
                - self.L * (self.L + 1)
            ) / (
                2 * self.J * (self.J + 1)
            )
        )

    def manifold(self, F):
        return HyperfineState(self, F)


@dataclass(frozen=True)
class HyperfineState:
    level: AtomicLevel
    F: sp.Expr

    @property
    def n(self):
        return self.level.n

    @property
    def L(self):
        return self.level.L

    @property
    def J(self):
        return self.level.J

    @property
    def I(self):
        return self.level.I

    @property
    def S(self):
        return self.level.S

    @property
    def gJ(self):
        return self.level.gJ

    @property
    def A(self):
        return self.level.hyperfine["A_MHz"]

    @property
    def B(self):
        return self.level.hyperfine["B_MHz"]

    @property
    def gF(self):
        if self.F == 0:
            return sp.S(0)
        else:
            return sp.simplify(
                self.gJ * (
                    self.F * (self.F + 1)
                    + self.J * (self.J + 1)
                    - self.I * (self.I + 1)
                ) / (
                    2 * self.F * (self.F + 1)
                )
            )

    def sublevels(self):
        return [
            MagneticSublevel(self, mF)
            for mF in range(
                -int(self.F),
                int(self.F) + 1
            )
        ]


@dataclass(frozen=True)
class MagneticSublevel:
    manifold: HyperfineState
    mF: sp.Expr

    @property
    def F(self):
        return self.manifold.F

    @property
    def J(self):
        return self.manifold.J

    @property
    def I(self):
        return self.manifold.I

    @property
    def L(self):
        return self.manifold.L

    @property
    def S(self):
        return self.manifold.S

    @property
    def n(self):
        return self.manifold.n

    @property
    def gJ(self):
        return self.manifold.gJ

    @property
    def gF(self):
        return self.manifold.gF

    @property
    def level(self):
        return self.manifold.level

    @property
    def atom(self):
        return self.manifold.level.atom