#ifndef SQUARE_LATTICE_H
#define SQUARE_LATTICE_H

#include <vector>
#include <random>

class SquareLattice {
public:
    SquareLattice(float interactionStrength, int latticeSize);
    
    std::vector<int>& get_lattice() { return lattice; }
    const std::vector<int>& get_lattice() const { return lattice; }
    void restore_random_lattice();
    
    float J;

private:
    std::vector<int> lattice;
    int size;
};

#endif // SQUARE_LATTICE_H