#%% Imports
from utilities import branching_rate, split_frequencies
from operators import Hamiltonian, Lindblad
from interactions import BField, EField
from rwa import RWA, optical_rotor
from densitymatrix import DensityMatrix
from atom import Atom
from parameters import build_num_values
from absorption import absorption_fast
from matrix_elements import electric_dipole_element

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import time
import pickle, os
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, coo_matrix
from scipy.signal import argrelextrema

import imageio.v2 as imageio

CACHE_FILE = "lindblad_cache.pkl"

#%% Symbols
ΩRF = sp.symbols('\u03A9RF', real=True)         # Rabi
ωRF = sp.symbols('\u03C9RF', positive=True)     # oscillation
ΩR = sp.symbols('\u03A9E', positive=True)       # Rabi
ωR = sp.symbols('\u03C9E', positive=True)       # oscillation
B0 = sp.symbols('B0', real=True)                # static field ampl
γ = sp.symbols('\u03B3', positive=True)         # gamma
Γ = sp.symbols('\u0393', positive=True)         # Gamma
BRF = sp.symbols('BRF', positive=True)
E0 = sp.symbols('E0', positive=True)

#%% Extract atomic data
cs_atom = Atom("data/cs.json")

#%% Numerical values
# BRF_values = [0.23, 0.46, 0.60, 0.80, 0.92, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
BRF_values = np.linspace(0.1, 5, 50)
E0_values = [2]
k = 9

zeeman_field = {
    'min_amplitude': 0,
    'max_amplitude': 5,
    'no_points': 300
    }

angles = {
    'E_theta': sp.pi/2,
    'E_phi': 0,
    'B_RF_theta': sp.pi/4,
    'B_RF_phi': 0
    }
states = {
    'ground_level': '6S1/2',
    'ground_manifold': 4,
    'excited_level': '6P1/2',
    'excited_manifold': 4
    }

B0_values = np.linspace(zeeman_field['min_amplitude'],
                        zeeman_field['max_amplitude'],
                        zeeman_field['no_points'])

colors = plt.colormaps['Reds'](np.linspace(0.2, 1, len(BRF_values)))

symbols_map = {"Gamma": Γ, "gamma": γ, "omega_RF": ωRF, "omega_R": ωR}
num_values = build_num_values(
    cs_atom,
    transition_name="D1",
    symbols=symbols_map,
    gamma_transit_MHz=0.015,     # apparatus: transit-time relaxation
    omega_RF_MHz=0.508,          # apparatus: RF drive frequency
    omega_R_MHz=0,               # optical detuning reference
    E1=0,
    E2=0,
)

#%% State creation
ground = cs_atom.atomic_level(
    states['ground_level']).manifold(states['ground_manifold'])
excited = cs_atom.atomic_level(
    states['excited_level']).manifold(states['excited_manifold'])

t0 = time.time()
print('Starting..')

ρ = DensityMatrix(ground=ground, excited=excited)

# ---- ONE-TIME precompute, right after ρ/excited_states/ground_states are defined ----
probe_field = EField(amplitude=E0_values[0], frequency=num_values[ωR], theta=sp.pi/2, phi=0)
dipole_table = np.zeros((len(ρ.excited_states), len(ρ.ground_states)), dtype=np.complex128)
for i, exc in enumerate(ρ.excited_states):
    for j, gnd in enumerate(ρ.ground_states):
        val = electric_dipole_element(probe_field, exc, gnd)
        dipole_table[i, j] = complex(sp.N(val))

#%% Build frequency split Lindbladian or simply load it
if os.path.exists(CACHE_FILE):
    print("Loading cached Lindbladian setup...")
    with open(CACHE_FILE, "rb") as f:
        frequncy_dict = pickle.load(f)
else:
    H = Hamiltonian(ρ)

    # Create interaction fields
    e_field = EField(amplitude=E0, frequency=ωR, theta=angles['E_theta'], phi=angles['E_phi'])
    b_rf_field = BField(amplitude=BRF, frequency=ωRF, theta=angles['B_RF_theta'], phi=angles['B_RF_phi'])
    b_zeeman_field = BField(amplitude=B0, frequency=0)

    H.add_field(e_field)
    H.add_field(b_rf_field)
    H.add_field(b_zeeman_field)

    # Optical RWA
    rotor = optical_rotor(ρ, ωR)
    H = RWA(H, rotor, ωR)

    print(f"Hamiltonian creation took {time.time()-t0:.1f}s")
    t0 = time.time()

    # Create Lindblad superoperator from the given Hamiltonian
    L = Lindblad(ρ, H)

    for exc in ρ.excited_states:
        total = sum(branching_rate(exc, gnd)
            for gnd in ρ.ground_states)
        for gnd in ρ.ground_states:
            rate = branching_rate(exc, gnd) / total
            L.spontaneous_decay(exc, gnd, rate, Γ)

    # transit loss
    L.add_transit(γ)

    A = L.get_coeff()

    print(f"Lindblad coefficients took {time.time()-t0:.1f}s")
    t0 = time.time()

    freq_mtx = split_frequencies(A, ωRF)
        
    frequncy_dict = {n: mtx.subs(num_values)
                for n, mtx in freq_mtx.items()}
    
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(frequncy_dict, f)
    
    print('Done building.')

r = sp.zeros((2*k + 1) * ρ.len**2, 1)
b = sp.zeros(ρ.len**2, 1)

for state in ρ.ground_states:
    i = ρ.states.index(state)
    b[i*ρ.len + i] -= num_values[γ] / len(ρ.ground_states)

r[k*ρ.len**2:(k+1)*ρ.len**2, 0] = b
r = np.array(r, dtype=np.complex128).flatten()
#%% Identify nonzero entries symbolically (ONCE), then lambdify each

nonzero_positions = {}      # order -> list of (i, j) that are ever nonzero
nonzero_exprs_flat = {}     # order -> flat list of the nonzero sympy expressions
A_func_batched = {}         # order -> single compiled function

for n, mtx in frequncy_dict.items():
    rows, cols = mtx.shape
    positions = [(i, j) for i in range(rows) for j in range(cols)
                 if mtx[i, j] != 0]
    nonzero_positions[n] = positions
    nonzero_exprs_flat[n] = [mtx[i, j] for (i, j) in positions]
    A_func_batched[n] = sp.lambdify((B0, BRF, E0), nonzero_exprs_flat[n],
                                 modules="numpy", cse=True)

nonzero_idx_arrays = {
    order: (np.array([p[0] for p in positions]),
            np.array([p[1] for p in positions]))
    for order, positions in nonzero_positions.items()}

def get_block_standalone(order, block_data_dict):
    """Builds a dense (block_dim, block_dim) array for a single order,
    given block_data_dict = {order: flat array of nonzero values}."""
    dense = np.zeros((block_dim, block_dim), dtype=np.complex128)
    if order not in block_data_dict:
        return dense
    ii, jj = nonzero_idx_arrays[order]
    dense[ii, jj] = block_data_dict[order]
    return dense

def eval_block_for_all_B0(order, B0_values, BRF_value, E0_value):
    """Returns array of shape (block_dim, block_dim, N). Only nonzero
    entries are computed; everything else stays zero."""
    N = len(B0_values)
    out = np.zeros((block_dim, block_dim, N), dtype=np.complex128)
    vals = A_func_batched[order](B0_values, BRF_value, E0_value)
    for (i, j), v in zip(nonzero_positions[order], vals):
        out[i, j, :] = np.broadcast_to(v, (N,))
    return out

print(f"Elementwise lambdify setup took {time.time()-t0:.1f}s")
t0 = time.time()

#%% Precompute Floquet sparsity structure using ONLY the nonzero entries
state_dim = ρ.len
block_dim = ρ.len**2
n_dim = 2*k + 1
order_set = set(A_func_batched.keys())

block_positions = [(row, col, row - col)
                    for row in range(n_dim) for col in range(n_dim)
                    if (row - col) in order_set]

rows_idx_list = []
cols_idx_list = []
block_offsets = []              # (start, end, order) into the flat data array for this block
diag_shift_pos_list, diag_shift_m_list = [], []

cursor = 0
for (row, col, order) in block_positions:
    positions = nonzero_positions[order]
    n_entries = len(positions)
    local_i = np.array([p[0] for p in positions])
    local_j = np.array([p[1] for p in positions])

    rows_idx_list.append(row*block_dim + local_i)
    cols_idx_list.append(col*block_dim + local_j)
    block_offsets.append((cursor, cursor + n_entries, order))

    if order == 0:
        diag_mask = local_i == local_j
        diag_shift_pos_list.append(cursor + np.where(diag_mask)[0])
        diag_shift_m_list.append(np.full(diag_mask.sum(), row - k))

    cursor += n_entries

rows_idx = np.concatenate(rows_idx_list)
cols_idx = np.concatenate(cols_idx_list)
total_nnz = cursor
diag_shift_pos = np.concatenate(diag_shift_pos_list) if diag_shift_pos_list else np.array([], dtype=int)
diag_shift_m = np.concatenate(diag_shift_m_list) if diag_shift_m_list else np.array([])
FLOQUET_SHAPE = (n_dim*block_dim, n_dim*block_dim)

def build_floquet_matrix_fast(block_data_dict, omega_rf_num):
    """block_data_dict: {order: flat array of nonzero values, matching
    nonzero_positions[order] ordering}"""
    data = np.empty(total_nnz, dtype=np.complex128)
    for start, end, order in block_offsets:
        data[start:end] = block_data_dict[order]
    if len(diag_shift_pos):
        data[diag_shift_pos] -= 1j * diag_shift_m * omega_rf_num
    return coo_matrix((data, (rows_idx, cols_idx)), shape=FLOQUET_SHAPE).tocsr()

def solve_block_tridiagonal(D0_base, L_sparse, U_block, omega_rf_num, r, n_dim, block_dim, k):
    diag_idx = np.arange(block_dim)
    D_blocks = []
    for row in range(n_dim):
        m = row - k
        Dm = D0_base.copy()
        Dm[diag_idx, diag_idx] -= 1j * m * omega_rf_num
        D_blocks.append(Dm)

    # L_sparse = csr_matrix(L_block)   # convert once per call — cheap given low nnz

    c_prime = [None]*n_dim
    d_prime = [None]*n_dim
    r_blocks = [r[i*block_dim:(i+1)*block_dim] for i in range(n_dim)]

    rhs0 = np.hstack([U_block, r_blocks[0][:, None]])
    sol0 = np.linalg.solve(D_blocks[0], rhs0)
    c_prime[0] = sol0[:, :-1]
    d_prime[0] = sol0[:, -1]

    for i in range(1, n_dim):
        Lc = L_sparse @ c_prime[i-1]       # sparse @ dense — fast
        Ld = L_sparse @ d_prime[i-1]       # sparse @ dense (vector) — fast

        denom = D_blocks[i] - Lc
        rhs_vec = r_blocks[i] - Ld

        if i < n_dim - 1:
            rhs = np.hstack([U_block, rhs_vec[:, None]])
            sol = np.linalg.solve(denom, rhs)
            c_prime[i] = sol[:, :-1]
            d_prime[i] = sol[:, -1]
        else:
            d_prime[i] = np.linalg.solve(denom, rhs_vec)

    x_blocks = [None]*n_dim
    x_blocks[-1] = d_prime[-1]
    for i in range(n_dim-2, -1, -1):
        x_blocks[i] = d_prime[i] - c_prime[i] @ x_blocks[i+1]   # this one stays dense (c_prime is dense)

    return np.concatenate(x_blocks)

print(f"Sparsity precompute took {time.time()-t0:.1f}s")
t0 = time.time()

#%%
def solve_one_B0(idx, B0_val, D0_base_all, L_sparse, U_block, omega_rf_num, r, n_dim, block_dim, k, state_dim, n_ground):
    D0_base = get_block_standalone(0, {0: D0_base_all[:, idx]})
    solution = solve_block_tridiagonal(D0_base, L_sparse, U_block, omega_rf_num, r, n_dim, block_dim, k)
    ρ_tmp = solution[k*state_dim**2:(k+1)*state_dim**2]
    ρ0 = ρ_tmp.reshape(state_dim, state_dim)   # much faster than the manual double loop too
    return absorption_fast(dipole_table, ρ0, num_values[Γ])

def compute_result_rf(BRF_value, E0_value, B0_values, resid_tol=1e-8):
    N = len(B0_values)
    omega_rf_num = num_values[ωRF]

    # vals_by_order[order] has shape (n_nonzero_for_order, N)
    vals_by_order = {}
    for order in order_set:
        raw = A_func_batched[order](B0_values, BRF_value, E0_value)
        vals_by_order[order] = np.array(
            [np.broadcast_to(v, (N,)) for v in raw], dtype=np.complex128
        )

    exc_populations = np.empty(N)

    L_block = get_block_standalone(-1, {-1: vals_by_order[-1][:, 0]})   # any column works if B0-independent
    U_block = get_block_standalone(1, {1: vals_by_order[1][:, 0]})
        
    L_sparse = csr_matrix(L_block)

    exc_populations = Parallel(n_jobs=3)(
        delayed(solve_one_B0)(idx, B0_values[idx], vals_by_order[0], L_sparse, U_block,
                               omega_rf_num, r, n_dim, block_dim, k, state_dim, len(ρ.ground_states))
            for idx in range(N)
    )
    exc_populations = np.array(exc_populations)

    return exc_populations, BRF_value

results = [compute_result_rf(BRF_value, E0_values[0], B0_values) for BRF_value in BRF_values]

print(f'Solve loop ({zeeman_field["no_points"]} B0 points) took {time.time()-t0:.1f}s')

#%% Plotting
peak_indeces = []
peak_splitting_145 = []
peak_splitting_290 = []

frames = []

for (exc_pop, BRF_value), color in zip(results, colors):
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.set_title(
        rf'$E \angle {{{angles["E_theta"]/np.pi*90:0.1f}}}$, '
        rf'$F={{{states["ground_manifold"]}}} \Leftrightarrow '
        rf'F={{{states["excited_manifold"]}}}$ '
        rf'$B_{{RF}}={{{BRF_value:0.1f}}}$'
        #rf'$E_0$ = {E0_values[0]:0.1f} V/m'
    )
    ax.label_outer()
    ax.set_box_aspect(0.5)
        
    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(right=0.75)
    fig.legend(handles, labels,
           loc="center left",
           bbox_to_anchor=(0.75, 0.5),
           ncol=1)
    ax.set_xlabel('Magnētiskā lauka intensitāte [G]')
    ax.set_ylabel('Absorbcijas signāls [a.u.]')
    ax.set_ylim(4.7, 5.65)
    
    peak_indeces = argrelextrema(exc_pop, np.greater)[0]
    for i in range(len(peak_indeces)):
        if B0_values[peak_indeces[i]] >= 1 and \
            B0_values[peak_indeces[i]] <= 2 and \
            B0_values[peak_indeces[i-1]] >= 1 and \
            B0_values[peak_indeces[i-1]] <= 2:
                peak_splitting_145.append([BRF_value, abs(
                    B0_values[peak_indeces[i]] - B0_values[peak_indeces[i-1]])])
        
        if B0_values[peak_indeces[i]] >= 2 and \
            B0_values[peak_indeces[i]] <= 3.5 and \
            B0_values[peak_indeces[i-1]] >= 2 and \
            B0_values[peak_indeces[i-1]] <= 3.5:
                peak_splitting_290.append([BRF_value, abs(
                    B0_values[peak_indeces[i]] - B0_values[peak_indeces[i-1]])])
            
    ax.plot(B0_values, exc_pop, color=color)#, \
    #        label=rf'$B_{{\text{{RF}}}}$={BRF_value*1000:0.1f}mG')
    
    fig.canvas.draw()

    frame = np.asarray(fig.canvas.buffer_rgba())
    frames.append(frame)

    plt.close(fig)
    
imageio.mimsave("animation_mixed.gif", frames, duration=0.3, loop=0)

#fig.canvas.draw()
#plt.show()

peak_splitting_145 = np.array(peak_splitting_145)
peak_splitting_290 = np.array(peak_splitting_290)

'''
plt.figure()
plt.scatter(peak_splitting_145[:,0],peak_splitting_145[:,1])
plt.title('1.45 šķelšanās')
plt.show()
'''

plt.figure()
plt.scatter(peak_splitting_290[:,0],peak_splitting_290[:,1])
plt.title('2.90 šķelšanās')
plt.show()