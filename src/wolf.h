#ifndef WOLFF_ALGORITHM_H
#define WOLFF_ALGORITHM_H

#include <vector>
#include <random>
#include <unordered_set>
#include <queue>
#include "SquareLattice.h"

class Wolff {
public:
    Wolff(float interactionStrength, int latticeSize, float T_MIN, float T_MAX, float T_STEP, long int IT);
    void simulate_phase_transition();
    void store_results_to_file() const;

    std::vector<float> Temperatures;
    std::vector<float> MagnetizationResults;
    std::vector<float> EnergyResults;
    std::vector<float> Susceptibility;
    std::vector<float> SpecificHeat;
    std::vector<float> BinderCumulant;

private:
    SquareLattice lattice;
    float T_MIN, T_MAX, T_STEP;
    int L, N;
    long int IT;
    std::mt19937 gen;
    std::uniform_real_distribution<float> dis;

    float calculate_magnetization_per_site(const std::vector<int>& lattice) const;
    float calculate_energy_per_site(const std::vector<int>& lattice) const;
    void add_to_cluster(std::vector<int>& lattice, std::unordered_set<int>& cluster, 
                       std::queue<int>& spin_queue, float P, int i);
    void update(std::vector<int>& lattice, float T, float J);
};

#endif
