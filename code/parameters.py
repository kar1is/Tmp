import numpy as np
import sympy as sp


def atomic_num_values(atom, transition_name, symbols):
    """
    symbols: dict mapping names used here to the sympy symbols
    defined in main.py, e.g. {"Gamma": Γ}
    """
    transition = atom.transition(transition_name)
    return {
        symbols["Gamma"]: 2*np.pi * transition["linewidth_MHz"],
    }


def experiment_num_values(symbols,
                          *,
                          gamma_transit_MHz,
                          omega_RF_MHz,
                          omega_R_MHz=0,
                          E1=0,
                          E2=0):
    return {
        symbols["gamma"]: 2*np.pi * gamma_transit_MHz,
        symbols["omega_RF"]: 2*np.pi * omega_RF_MHz,
        symbols["omega_R"]: 2*np.pi * omega_R_MHz,
        sp.symbols("E1"): E1,
        sp.symbols("E2"): E2,
    }


def build_num_values(atom, transition_name, symbols, **experiment_kwargs):
    values = {}
    values.update(atomic_num_values(atom, transition_name, symbols))
    values.update(experiment_num_values(symbols, **experiment_kwargs))
    return values