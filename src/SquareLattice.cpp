#include "SquareLattice.h"
#include <random>
#include <algorithm>

SquareLattice::SquareLattice(float interactionStrength, int latticeSize) 
    : J(interactionStrength), size(latticeSize) {
    lattice.resize(latticeSize * latticeSize);
    restore_random_lattice();
}

void SquareLattice::restore_random_lattice() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    
    for (auto& spin : lattice) {
        spin = dis(gen) ? 1 : -1;
    }
}