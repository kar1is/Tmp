from dataclasses import dataclass, field
from symbols import t
import sympy as sp


@dataclass
class Field:
    amplitude: sp.Expr
    frequency: sp.Expr
    theta: sp.Expr = sp.Integer(0)
    phi: sp.Expr = sp.Integer(0)

    symbols: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self._add_sym([self.amplitude, self.frequency])

    def _add_sym(self, symbols):
        if not isinstance(symbols, (list, tuple)):
            symbols = [symbols]
        for symbol in symbols:
            if isinstance(symbol, sp.Symbol) and symbol not in self.symbols:
                self.symbols.append(symbol)

    @property
    def x(self):
        return self.amplitude * sp.sin(self.theta) * sp.cos(self.phi) * \
            sp.Rational(1,2) * (sp.exp(-sp.I * self.frequency * t) + 
                                sp.exp(sp.I * self.frequency * t))
    @property
    def y(self):
        return self.amplitude * sp.sin(self.theta) * sp.sin(self.phi) * \
            sp.Rational(1,2) * (sp.exp(-sp.I * self.frequency * t) + 
                                sp.exp(sp.I * self.frequency * t))
    @property
    def z(self):
        return self.amplitude * sp.cos(self.theta) * \
            sp.Rational(1,2) * (sp.exp(-sp.I * self.frequency * t) + 
                                sp.exp(sp.I * self.frequency * t))


@dataclass
class BField(Field):
    """
    Magnetic field.

    amplitude:
        Physical magnetic-field amplitude, e.g. in Gauss.

    frequency:
        Angular frequency of the field.

    theta, phi:
        Direction of the field relative to the quantization axis.
    """
    pass


@dataclass
class EField(Field):
    """
    Electric field.

    amplitude:
        Physical electric-field amplitude.

    frequency:
        Angular frequency.

    theta, phi:
        Direction relative to the quantization axis.
    """
    # def __post_init__(self):
    #     self.amplitude *= 0.0804
    pass