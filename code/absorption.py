import sympy as sp
import numpy as np

from matrix_elements import electric_dipole_element


def absorption(excited_states, ground_states, rho, Gamma, field):
    Gamma_num = float(sp.N(Gamma))
    linewidth = Gamma_num

    omega_probe = 0.0
    omega_eig = np.zeros(len(excited_states), dtype=float)

    denominator = (omega_probe - omega_eig)**2 + 0.25 * linewidth**2

    contributions = []
    
    rho_num = np.asarray(
        [[complex(sp.N(rho[j, k])) for k in range(rho.shape[1])]
         for j in range(rho.shape[0])],
        dtype=np.complex128)

    for i, excited in enumerate(excited_states):

        for j, ground_j in enumerate(ground_states):
            for kk, ground_k in enumerate(ground_states):

                rho_jk = rho_num[j, kk]

                # Electric dipole matrix element
                A = electric_dipole_element(field, excited, ground_j)

                # Second matrix element
                B = electric_dipole_element(field, excited, ground_k)

                A_num = complex(sp.N(A))
                B_num = complex(sp.N(sp.conjugate(B)))

                contribution = A_num * rho_jk * B_num / \
                                float(denominator[i])

                contributions.append(float(np.real(contribution)))
    return contributions

def absorption_fast(dipole_table, rho_num, Gamma_num):
    """
    dipole_table: (n_excited, n_ground) complex array, precomputed once.
    rho_num: (dim, dim) complex numpy array (the solved density matrix).
    Returns the same summed signal as before, but numerically.
    """
    n_exc, n_gnd = dipole_table.shape
    n_ground_states = n_gnd
    # rho block corresponding to ground-ground coherences
    rho_gg = rho_num[:n_gnd, :n_gnd]
    # signal = sum_i sum_j sum_k  A_ij * rho_jk * conj(A_ik)
    # = sum_i  dipole_table[i,:] @ rho_gg @ conj(dipole_table[i,:])
    signal = 0.0
    for i in range(n_exc):
        v = dipole_table[i, :]
        signal += np.real(v @ rho_gg @ np.conj(v))
    return signal