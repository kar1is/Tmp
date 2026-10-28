import utilities as ut
from operators import Hamiltonian, Lindblad, Project
from interactions import BField, EField
from rwa import RWA
from densitymatrix import DensityMatrix
from symbols import t

from joblib import Parallel, delayed
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import imageio.v2 as imageio

np.set_printoptions(precision=2, suppress=True)

omegaB_RF = sp.symbols('\u03A9RF', positive=True)
omegaB_RF_small = sp.symbols('\u03C9RF', positive=True)
omega_laser_small = sp.symbols('\u03C9E', positive=True)
B0 = sp.symbols('B0')
rate_trans_rlx = sp.symbols('\u03B3', positive=True)
rate_spont_rlx = sp.symbols('\u0393', positive=True)
omega_laser = sp.symbols('\u03A9E', positive=True)

num_values = {
    rate_trans_rlx: 0.01, #gamma
    rate_spont_rlx: 4.575, #Gamma
    sp.symbols('E1'): 0,
    sp.symbols('E2'): 0,
    sp.symbols('E3'): 0,
    sp.symbols('E4'): 0,
    sp.symbols('E5'): 0,
    sp.symbols('E6'): 0,
    sp.symbols('E7'): 0,
    sp.symbols('E8'): 0,
    sp.symbols('E9'): 0,
    sp.symbols('E10'): 0,
    sp.symbols('E11'): 0,
    sp.symbols('E12'): 0,
    sp.symbols('E13'): 0,
    sp.symbols('E14'): 0,
    sp.symbols('E15'): 0,
    sp.symbols('E16'): 0,
    omegaB_RF_small: 0.508, # RF freq # 508kHz
    # omegaB_RF: 2*np.pi*0.1, # RF Rabi freq
    omega_laser_small: 0, # E freq
    # sp.symbols('\u03A9E', positive=True): 1, # E Rabi freq
    # 0.1 - 10 MHz
    # (0.1, 1, 10) x 2 konfiguracijas
}

# Create density matrix
ground_states = ut.create_states(n=6, L=0, F=1)
excited_states = ut.create_states(n=6, L=1, F=0)
total_states = ground_states + excited_states
rho = DensityMatrix(total_states)

# Create interaction fields
efield = EField(1, omega_laser_small, 'pi')
# efield = EField(1, omega_laser_small, 'x') # for E \perp B conf
bfield_rf = BField(1, omegaB_RF_small, 'x')
bfield_z = BField(1, 0, 'pi')
bfield_z_ampl = np.linspace(-3, 3, 500)

H = Hamiltonian(rho)
H.add_field(efield)
H.add_field(bfield_z)
H.add_field(bfield_rf)

rotor1 = sp.eye(rho.len)
'''
for i in range(rho.len):
    if rho.states[i].L == 0:
        rotor1[i,i] = sp.srepr(sp.exp((-1)**abs(rho.states[i].mF) * sp.Rational(1,2) * sp.I * \
                                          omegaB_RF_small * t))
    if rho.states[i].L == 1:
        rotor1[i,i] = sp.srepr(sp.exp((-1)**abs(rho.states[i].mF) * sp.Rational(1,2) * sp.I * \
                                          omegaB_RF_small * t))
'''
rotor1[0,0] = sp.srepr(sp.exp(sp.I * omegaB_RF_small * t))
rotor1[2,2] = sp.srepr(sp.exp((-1)*sp.I * omegaB_RF_small * t))
H = RWA(H, rotor1, omegaB_RF_small)
# print('Rotor 1:')
# sp.pprint(rotor1)
# print('HAMILTONIAN\n\n')
# sp.pprint(H.mtx)
# print('\n\n\n')

rotor2 = sp.eye(rho.len)
rotor2[0,0] = sp.srepr(sp.exp(sp.I * omegaB_RF_small * t))
rotor2[2,2] = sp.srepr(sp.exp(sp.I * omegaB_RF_small * t))
rotor2[3,3] = sp.srepr(sp.exp(sp.I * omegaB_RF_small * t))
# H = RWA(H, rotor2, omegaB_RF_small)

rotor_perp = sp.eye(rho.len)
rotor_perp[-1,-1] = sp.srepr(sp.exp(sp.I * omegaB_RF_small * t))
# H = RWA(H, rotor_perp, omegaB_RF_small)

rotor3 = sp.eye(rho.len)
for i in range(rho.len):
    if rho.states[i].L == 1: 
        rotor3[i,i] = sp.srepr(sp.exp((-1)*sp.I * omega_laser_small * t))
# print('Rotor 3:')
# sp.pprint(rotor3)
H = RWA(H, rotor3, omega_laser_small)

print('Hamiltonian:')
sp.pprint(H.mtx)

lindbladian = Lindblad(H)

relax = [] # Symbolic jump operators
for i in range(rho.len-1):
    tmp_op = Project(rho, i, -1, rate_spont_rlx)
    lindbladian.add_jump(tmp_op)
    tmp_op = Project(rho, i, i, rate_trans_rlx)
    relax.append(tmp_op)
    lindbladian.add_jump(tmp_op)

coeff = lindbladian.get_coeff()
for i in range(rho.len**2):
    coeff[-1,i] = 0
for i in range(1, rho.len+1):
    coeff[-1,rho.len*(i-1)+i-1] = 1

b = np.zeros(rho.len**2)
b[-1] = 1

# sp.pprint(lindbladian.mtx)
# sp.pprint(sp.ccode(H.mtx))

omegas_rf = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1] # np.linspace(0.001, 0.1, 5)
# omegas_rf = [0.001, 0.01, 0.05, 0.1]
# omegas_rf = np.linspace(0.001, 0.5, 100)
colors = plt.colormaps['Reds'](np.linspace(0.2, 1, len(omegas_rf)))
e_fields = np.linspace(0.1, 5, 30)

coeff = coeff.subs(num_values)
coeff_func = sp.lambdify((B0, omegaB_RF, omega_laser), coeff, "numpy")

def compute_result(e_field):
    exc_pop = []
    max_v = 0
    for i in range(len(omegas_rf)):
        tmp_sol = []
        for strength in bfield_z_ampl:
            tmp_coeff = np.asarray(coeff_func(strength, 
                                              omegas_rf[i], e_field),
                                   dtype=np.complex128)
            solution = np.linalg.solve(tmp_coeff, b)
            tmp_sum = 0
            for qq in range(len(ground_states)+1, rho.len+1):
                tmp_sum += abs(solution[rho.len*(qq-1)+qq-1])
            tmp_sol.append(tmp_sum)
        max_v = max(tmp_sol)
        exc_pop.append(tmp_sol)
    for i in range(len(exc_pop)):
        exc_pop[i] /= max_v
    return exc_pop, e_field

results = Parallel(n_jobs=4)(
    delayed(compute_result)(e)
    for e in e_fields
)

def compute_result_rf(omega_rf):
    tmp_sol = []
    for strength in bfield_z_ampl:
        tmp_coeff = np.asarray(coeff_func(strength, 
                                          omega_rf, e_fields[0]),
                               dtype=np.complex128)
        solution = np.linalg.solve(tmp_coeff, b)
        tmp_sum = 0
        for qq in range(len(ground_states)+1, rho.len+1):
            tmp_sum += abs(solution[rho.len*(qq-1)+qq-1])
        tmp_sol.append(tmp_sum)
    tmp_sol /= max(tmp_sol)
    return tmp_sol, omega_rf

'''
results = Parallel(n_jobs=4)(
    delayed(compute_result_rf)(omega_rf)
    for omega_rf in omegas_rf
)
'''

frames = []

for exc_pop, e_field in results:
#for exc_pop, omega_rf in results:
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    for i in range(len(exc_pop)):  
        ax.plot(bfield_z_ampl, exc_pop[i], color=colors[i], \
                   label=rf'$\Omega_{{\text{{RF}}}}$={omegas_rf[i]*1000}kHz')

    # ax.plot(bfield_z_ampl, exc_pop, color=colors[-1])
    ax.set_title(rf'$E \perp B_0$, $F=1 \Leftrightarrow F=0$, $\Omega_R$ = {round(e_field,2)}MHz')
    ax.label_outer()
    # ax.set_title(rf'$E \| B_0$, $F=1 \Leftrightarrow F=0$, $\Omega_R$ = {round(e_fields[0],2)}MHz, $\Omega_{{RF}}$ = {round(omega_rf,2)*1000}kHz')
    ax.set_box_aspect(0.5)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(right=0.75)
    fig.legend(handles, labels,
           loc="center left",
           bbox_to_anchor=(0.75, 0.5),
           ncol=1)
    ax.set_xlabel('Magnētiskā lauka intensitāte [G]')
    ax.set_ylabel('Ierosinātā stāvokļa apdzīvotība [a.u.]')
    # ax.set_xlabel('Megnetic field intensity [G]')
    # ax.set_ylabel('Excited state population [a.u.]')
    
    fig.canvas.draw()

    frame = np.asarray(fig.canvas.buffer_rgba())
    frames.append(frame)

    plt.close(fig)
    
imageio.mimsave("animation_perp_diff.gif", frames, duration=0.3, loop=0)

'''
    if len(e_fields) > 1:
        
    else:
        ax.plot(bfield_z_ampl, exc_pop[0] / max_v, color=colors[0], \
                   label=rf'$\Omega_{{\text{{RF}}}}$={omegas_rf[0]*1000}kHz')
        ax.set_title(rf'$\Omega_R$ = {e_fields[0]}MHz')
        ax.label_outer()
        ax.set_box_aspect(0.5)
'''

'''
if len(e_fields) > 1:
    handles, labels = ax[0].get_legend_handles_labels()
else:
    handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels,
           loc="center left",
           bbox_to_anchor=(1.005, 0.5),
           ncol=1)
fig.supxlabel('Magnētiskā lauka intensitāte [G]')
fig.supylabel('Ierosinātā stāvokļa apdzīvotība [a.u.]')
plt.tight_layout()
# fig.set_constrained_layout(True)
'''
plt.savefig("plot2_smth.png", dpi=300, bbox_inches="tight")
plt.show()