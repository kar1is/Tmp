#include<vector>

struct atomic_state {
    int L;
    double F, mF;
};

std::vector<atomic_state> build_states(int L, double F) {
    int state_count = 2*(int)F+1;
    std::vector<atomic_state> atomic_states(state_count);
    for(int i=0; i<state_count; i++) {
        atomic_states[i].L = L;
        atomic_states[i].F = F;
        atomic_states[i].mF = (-1)*F + (double)i;
    }
    return atomic_states;
}

std::vector<std::vector<double>> add_B_field(
        std::vector<std::vector<double>> hamiltonian,
        std::vector<atomic_state> atomic_states,
        double amp_B,
        double pol_B
        ) {
    int n = sizeof(hamiltonian[0]) / sizeof(hamiltonian[0][0]);
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            if(pol_B == 'pi')
        }
    }
    return hamiltonian;
}

int main() {

    // Create atomic state matrix
    const int ground_L = 0;        
    const int ground_F = 4;        
    const std::vector<atomic_state> ground_states = build_states(ground_L, ground_F);
    const int excited_L = 1;        
    const int excited_F = 4;        
    const std::vector<atomic_state> excited_states = build_states(excited_L, excited_F);
    std::vector<atomic_state> atomic_states;
    atomic_states.insert(atomic_states.end(), ground_states.begin(), ground_states.end());
    atomic_states.insert(atomic_states.end(), ground_states.begin(), ground_states.end());

    
    return 0;
}
