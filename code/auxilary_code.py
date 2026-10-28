H_coeff = split_frequencies(H.mtx, ωRF)
H_coeff = {n: mtx.subs(num_values) for n, mtx in H_coeff.items()}

H_func = {
    n: sp.lambdify((B0, ΩRF, ΩR), mtx, "numpy")
    for n, mtx in H_coeff.items()
}

def build_hamiltonian_floquet(Hblocks, omega_rf, k):
    dim = Hblocks[0].shape[0]
    n_dim = 2*k + 1
    HF = np.zeros((n_dim*dim, n_dim*dim), dtype=complex)
    for row in range(n_dim):
        m = row - k
        for col in range(n_dim):
            order = row - col
            if order not in Hblocks:
                continue
            r0, c0 = row*dim, col*dim
            HF[r0:r0+dim, c0:c0+dim] = Hblocks[order]
            if order == 0:
                HF[r0:r0+dim, c0:c0+dim] += np.eye(dim) * m * omega_rf
    return HF

def diagonalize(B0, ΩRF_val, ΩR_val, k, dim):
    blocks = {n: H_func[n](B0, ΩRF_val, ΩR_val) for n in H_func}
    HF = build_hamiltonian_floquet(blocks, num_values[ωRF], k)
    evals, evecs = np.linalg.eigh(HF)   # eigh: sorted, returns eigenvectors too
    return evals / (2*np.pi), evecs

# ---------------------------------------------------------------
# STEP 1: fine min-gap scan to locate + size the avoided crossing
# ---------------------------------------------------------------
def min_gap_scan(B0_range, ΩRF_val, ΩR_val, k, dim, n_points=500):
    B0_fine = np.linspace(*B0_range, n_points)
    min_gaps = np.empty(n_points)
    all_evals = np.empty((n_points, (2*k+1)*dim))
    for i, B0 in enumerate(B0_fine):
        ev, _ = diagonalize(B0, ΩRF_val, ΩR_val, k, dim)
        all_evals[i] = ev
        min_gaps[i] = np.min(np.diff(np.sort(ev)))
    return B0_fine, min_gaps, all_evals

dim = ρ.len
k_check = 6              # bump up for a truncation-stability check
ΩRF_val = 2*np.pi*0.1
ΩR_val  = 2*np.pi*1

B0_fine, min_gaps, all_evals = min_gap_scan(
    (0, 1), ΩRF_val, ΩR_val, k_check, dim, n_points=1000
)

plt.plot(B0_fine, min_gaps)
plt.xlabel('B0 [G]'); plt.ylabel('Min quasienergy gap [MHz]')
plt.show()

# locate local minima (candidate avoided crossings)
from scipy.signal import argrelextrema
minima_idx = argrelextrema(min_gaps, np.less)[0]
for idx in minima_idx:
    print(f"B0 = {B0_fine[idx]:.5f} G,  gap = {min_gaps[idx]*1e3:.4f} kHz")

# ---------------------------------------------------------------
# STEP 2: truncation-stability check — repeat at larger k
# ---------------------------------------------------------------
for k_test in [4, 6, 8, 10]:
    _, min_gaps_k, _ = min_gap_scan((0, 1), ΩRF_val, ΩR_val, k_test, dim, n_points=200)
    print(f"k={k_test}: min gap = {min_gaps_k.min()*1e3:.4f} kHz")
# if the min gap converges (stops changing) as k grows -> real feature
# if it keeps drifting/changing a lot -> truncation artifact

# ---------------------------------------------------------------
# STEP 3: eigenvector-based photon-order identification
# ---------------------------------------------------------------
def identify_crossing(B0, ΩRF_val, ΩR_val, k, dim):
    ev, evecs = diagonalize(B0, ΩRF_val, ΩR_val, k, dim)
    order = np.argsort(ev)
    ev_sorted = ev[order]
    evecs_sorted = evecs[:, order]

    gaps = np.diff(ev_sorted)
    i = np.argmin(gaps)          # index of the near-degenerate pair
    vec_a = evecs_sorted[:, i]
    vec_b = evecs_sorted[:, i+1]

    n_dim = 2*k + 1
    # dominant Floquet block (sideband index) for each eigenvector
    def dominant_floquet_index(vec):
        weights = np.array([
            np.sum(np.abs(vec[block*dim:(block+1)*dim])**2)
            for block in range(n_dim)
        ])
        return np.argmax(weights) - k, weights  # -k shifts to true Floquet index m

    m_a, w_a = dominant_floquet_index(vec_a)
    m_b, w_b = dominant_floquet_index(vec_b)

    print(f"Gap = {gaps[i]*1e3:.4f} kHz at B0={B0}")
    print(f"State A: dominant Floquet index m = {m_a}, weight={w_a.max():.3f}")
    print(f"State B: dominant Floquet index m = {m_b}, weight={w_b.max():.3f}")
    print(f"Floquet index difference |Δm| = {abs(m_a - m_b)}  <- effective photon order")
    return m_a, m_b, w_a, w_b

def identify_crossing_mF(B0, ΩRF_val, ΩR_val, k, dim, ground_states):
    ev, evecs = diagonalize(B0, ΩRF_val, ΩR_val, k, dim)
    order = np.argsort(ev)
    ev_sorted = ev[order]; evecs_sorted = evecs[:, order]
    gaps = np.diff(ev_sorted)
    i = np.argmin(gaps)
    n_dim = 2*k + 1
    for label, vec in [("A", evecs_sorted[:, i]), ("B", evecs_sorted[:, i+1])]:
        weights = np.abs(vec)**2
        # collapse Floquet blocks, keep only physical-state weight
        state_weight = np.zeros(dim)
        for block in range(n_dim):
            state_weight += weights[block*dim:(block+1)*dim]
        top_state = np.argmax(state_weight)
        mF = ground_states[top_state].mF if top_state < len(ground_states) else "excited"
        print(f"{label}: dominant physical state idx={top_state} (mF={mF}), weight={state_weight[top_state]:.3f}")
    print(f"gap = {gaps[i]*1e3:.4f} kHz")

# identify_crossing(2.9165, ΩRF_val, ΩR_val, k_check, dim)
# identify_crossing(2.916, ΩRF_val, ΩR_val, k_check, dim)
# identify_crossing(2.918, ΩRF_val, ΩR_val, k_check, dim)
identify_crossing_mF(2.91562, ΩRF_val, ΩR_val, k_check, dim, ground_states)
identify_crossing_mF(2.91697, ΩRF_val, ΩR_val, k_check, dim, ground_states)
identify_crossing_mF(2.95511, ΩRF_val, ΩR_val, k_check, dim, ground_states)
identify_crossing_mF(2.96727, ΩRF_val, ΩR_val, k_check, dim, ground_states)

'''
'''
def find_real_crossings(B0_range, ΩRF_val, ΩR_val, k, dim, ground_states, n_points=200):
    B0_fine = np.linspace(*B0_range, n_points)
    results = []
    for B0 in B0_fine:
        ev, evecs = diagonalize(B0, ΩRF_val, ΩR_val, k, dim)
        order = np.argsort(ev)
        ev_sorted = ev[order]; evecs_sorted = evecs[:, order]
        gaps = np.diff(ev_sorted)
        i = np.argmin(gaps)
        n_dim = 2*k + 1
        def dominant_mF(vec):
            w = np.abs(vec)**2
            sw = np.zeros(dim)
            for block in range(n_dim):
                sw += w[block*dim:(block+1)*dim]
            idx = np.argmax(sw)
            return (ground_states[idx].mF if idx < len(ground_states) else 'exc'), sw[idx]
        mA, wA = dominant_mF(evecs_sorted[:, i])
        mB, wB = dominant_mF(evecs_sorted[:, i+1])
        results.append((B0, gaps[i], mA, mB, wA, wB))
    return results

res = find_real_crossings((2.9, 2.95), ΩRF_val, ΩR_val, k, dim, ground_states)
for B0, gap, mA, mB, wA, wB in res:
    if mA != mB and wA > 0.3 and wB > 0.3:
        print(f"B0={B0:.4f}  gap={gap*1e3:.4f}kHz  {mA}<->{mB}")
'''

'''
def build_hamiltonian_floquet(Hblocks, omega_rf, k):
    dim = Hblocks[0].shape[0]

    HF = np.zeros(((2*k+1)*dim,
                   (2*k+1)*dim), dtype=complex)

    for row in range(2*k+1):
        m = row-k

        for col in range(2*k+1):

            order = row-col

            if order not in Hblocks:
                continue

            r0 = row*dim
            c0 = col*dim

            HF[r0:r0+dim, c0:c0+dim] = Hblocks[order]

            if order == 0:
                HF[r0:r0+dim, c0:c0+dim] += np.eye(dim)*m*omega_rf

    return HF

B_res = 1.452  # from the resonance condition
blocks = {n: H_func[n](B_res, ΩRF_values[0], ΩR_values[0]) for n in H_func}
HF_res = build_hamiltonian_floquet(blocks, num_values[ωRF], 1)
ev = np.sort(np.linalg.eigvalsh(HF_res)) / (2*np.pi)
print(np.diff(ev))

kH = 3
energies = []

for B in B0_values:

    blocks = {
        n: H_func[n](B, ΩRF_values[0], ΩR_values[0])
        for n in H_func
    }

    HF = build_hamiltonian_floquet(
        blocks,
        num_values[ωRF],
        kH
    )

    ev = np.linalg.eigvalsh(HF)/(2*np.pi)
    energies.append(ev)
    
    min_gaps = [np.min(np.diff(np.sort(e))) for e in energies]
    
energies = np.array(energies)

plt.figure(figsize=(7,5))

plt.plot(B0_values, min_gaps)
plt.show()
#plt.xlim(2.9, 2.94)

for i in range(energies.shape[1]):
    plt.plot(B0_values, energies[:,i], 'k', lw=0.8)
plt.xlabel(r'$B_0$ [G]')
plt.ylabel('Quasienergy [MHz]')

# plt.xlim(2, 2.5)
plt.ylim(-0.5, 0.5)
plt.show()










# Add relaxation processes
# TODO: ADD MORE GENERAL RELAXATION
for i in range(ρ.len-1):
    # Spontaneous relaxation
    tmp_op = Project(ρ, i, ρ.len-1, Γ / len(ground_states))
    lindbladian.add_jump(tmp_op)
    # Transit relaxation
    tmp_op = Project(ρ, i, i, γ)
    lindbladian.add_jump(tmp_op)
    
for i in range(ρ.len):
    # Transit depopulation
    lindbladian.mtx[i,i] -= γ * ρ.list[ρ.len*i+i]

coeff = lindbladian.get_coeff() # Lindbladian coefficients

freq_mtx = split_frequencies(coeff, ωRF)

with open("output.txt", "w") as f:
    print('Hamiltonian:', file=f)
    with redirect_stdout(f):
        sp.pprint(H.mtx, wrap_line=False)
    print('Unsplit Lindbladian:', file=f)
    with redirect_stdout(f):
        sp.pprint(lindbladian.mtx, wrap_line=False)
    print("Split Lindbladian:", file=f)
    for freq, mtx in freq_mtx.items():
        print(f"\nFrequency component n = {freq}", file=f)
        with redirect_stdout(f):
            sp.pprint(mtx, wrap_line=False)

#%% Source (repopulation) term
r = np.zeros((2*k+1) * ρ.len**2)
b = np.zeros(ρ.len**2)

for i in range(len(ground_states)):
    b[ρ.len*i+i] = -num_values[γ] / len(ground_states)

r[k*ρ.len**2 : (k+1)*ρ.len**2] = b

#%% Numerical substitutions
freq_mtx = {n: mtx.subs(num_values) for n, mtx in freq_mtx.items()}

coeff_func = {n: sp.lambdify((B0, ΩRF, ΩR), mtx, "numpy")
              for n, mtx in freq_mtx.items()}


def build_floquet_matrix(freq_mtx, ωRF, k):
    blocks = [[None for _ in range(2*k+1)]
                    for _ in range(2*k+1)]
    for row in range(2*k+1):
        m = row - k
        for col in range(2*k+1):
            order = row - col

            if order not in freq_mtx:
                continue

            block = lil_matrix(freq_mtx[order])

            if order == 0:
                block = block - 1j*m*ωRF * \
                    identity(block.shape[0], format="lil")
            blocks[row][col] = block.tocsr()
    LF = bmat(blocks, format='csr')
    return LF


def solve_floquet(freq_mtx, ωRF, k, source_vec, tol=1e-10):
    LF = build_floquet_matrix(freq_mtx, ωRF, k)
    # x, info = gmres(LF, source_vec, rtol=tol,
    #                 atol=1e-12, restart=200, maxiter=1000)
    x = spsolve(LF.tocsc(), source_vec)
    return x

'''
#%%
def compute_result_rf(ΩRF, e_field, B0_values):
    exc_populations = []
    for strength in B0_values:
        numeric_freq_mtx = {n: func(strength, ΩRF, e_field)
                            for n, func in coeff_func.items()}
        solution = solve_floquet(numeric_freq_mtx, num_values[ωRF], k, r)
        
        ρ0 = solution[k*ρ.len**2 : (k+1)*ρ.len**2]
        tmp_exc_population = 0
        for i in range(len(ground_states), ρ.len):
            tmp_exc_population += ρ0[ρ.len*i+i]
        exc_populations.append(tmp_exc_population)
    return exc_populations, ΩRF

#%% Run calculation
results = Parallel(n_jobs=4)(
    delayed(compute_result_rf)(ΩRF, ΩR_values[0], B0_values)
    for ΩRF in ΩRF_values
)


'''