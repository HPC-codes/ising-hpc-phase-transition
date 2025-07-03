#ifndef WOLFF_ALGORITHM_H
#define WOLFF_ALGORITHM_H

#include <vector>
#include <random>
#include <unordered_set>
#include <queue>
#include <fstream>
#include <string>
#include "/content/ising-hpc-phase-transition/src/SquareLattice.h"

class Wolff {
public:
    Wolff(float interactionStrength, int latticeSize, float T_MIN, float T_MAX, float T_STEP, long int IT);
    ~Wolff();
    
    void simulate_phase_transition(bool save_time_evolution = false);
    void store_results_to_file() const;
    
    // Métodos para guardado de datos
    void save_time_step_data(const std::vector<int>& lattice, float T, int step);

    // Vectores de resultados
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
    mutable std::ofstream time_evolution_file;  // Archivo CSV para evolución temporal
    
    // Métodos de cálculo
    float calculate_magnetization_per_site(const std::vector<int>& lattice) const;
    float calculate_energy_per_site(const std::vector<int>& lattice) const;
    
    // Métodos del algoritmo Wolff
    void add_to_cluster(std::vector<int>& lattice, std::unordered_set<int>& cluster, 
                       std::queue<int>& spin_queue, float P, int i);
    void update(std::vector<int>& lattice, float T, float J);
    
    // Helper para CSV
    void initialize_time_evolution_file(float T);
};

#endif
