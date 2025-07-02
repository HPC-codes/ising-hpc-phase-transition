#ifndef WOLFF_ALGORITHM_H
#define WOLFF_ALGORITHM_H

#include <vector>
#include <random>
#include <iostream>
#include <unordered_set>
#include <queue>
#include <string>
#include "SquareLattice.h"

class Wolff {
public:
    Wolff(float interactionStrength, int latticeSize, float T_MIN, float T_MAX, float T_STEP, long int IT);
    void simulate_phase_transition();
    void store_results_to_file() const;

    // Métricas físicas
    std::vector<float> Temperatures;
    std::vector<float> MagnetizationResults;
    std::vector<float> EnergyResults;
    std::vector<float> Susceptibility;
    std::vector<float> SpecificHeat;
    std::vector<float> BinderCumulant;

protected:
    void add_to_cluster(std::vector<int>& lattice, std::unordered_set<int>& cluster, 
                       std::queue<int>& spin_queue, float P, int i);
    void update(std::vector<int>& lattice, float T, float J);
    float calculate_magnetization_per_site(const std::vector<int>& lattice);
    float calculate_energy_per_site(const std::vector<int>& lattice);

private:
    SquareLattice lattice;
    float T_MIN;
    float T_MAX;
    float T_STEP;
    int L;
    int N;
    long int IT;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;
};

#endif // WOLFF_ALGORITHM_H