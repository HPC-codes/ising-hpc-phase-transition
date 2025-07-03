#include "wolf.h"
#include <fstream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <iostream>
 
Wolff::Wolff(float interactionStrength, int latticeSize, float T_MIN, float T_MAX, float T_STEP, long int IT):
    lattice(interactionStrength, latticeSize),
    T_MIN(T_MIN), T_MAX(T_MAX), T_STEP(T_STEP),
    L(latticeSize), N(latticeSize * latticeSize), IT(IT),
    gen(std::random_device{}()), dis(0.0f, 1.0f)
{
    if (L <= 0) throw std::invalid_argument("Lattice size must be positive");
    if (T_MIN <= 0.0f || T_MAX <= 0.0f) throw std::invalid_argument("Temperature must be positive");
    if (T_STEP <= 0.0f) throw std::invalid_argument("Temperature step must be positive");
    if (IT <= 0) throw std::invalid_argument("Number of iterations must be positive");

    int expected_points = static_cast<int>((T_MAX-T_MIN)/T_STEP) + 1;
    Temperatures.reserve(expected_points);
    MagnetizationResults.reserve(expected_points);
    EnergyResults.reserve(expected_points);
    Susceptibility.reserve(expected_points);
    SpecificHeat.reserve(expected_points);
    BinderCumulant.reserve(expected_points);
}

Wolff::~Wolff() {
    if (time_evolution_file.is_open()) {
        time_evolution_file.close();
    }
}

void Wolff::initialize_time_evolution_file(float T) {
    if (!time_evolution_file.is_open()) {
        std::string filename = "ising_L" + std::to_string(L) + "_T" + 
                             std::to_string(T).substr(0, 4) + ".csv";
        time_evolution_file.open(filename);
        time_evolution_file << "Step,Temperature,Magnetization,Energy,LatticeConfig\n";
        time_evolution_file << std::fixed << std::setprecision(6);
    }
}

void Wolff::save_time_step_data(const std::vector<int>& lattice, float T, int step) {
    initialize_time_evolution_file(T);
    
    float M = calculate_magnetization_per_site(lattice);
    float E = calculate_energy_per_site(lattice);
    
    time_evolution_file << step << "," << T << "," << M << "," << E << ",\"";
    
    // Guardar configuración de red como cadena (+/-)
    for (size_t i = 0; i < lattice.size(); ++i) {
        time_evolution_file << (lattice[i] > 0 ? "+" : "-");
        if (i != lattice.size() - 1) time_evolution_file << " ";
    }
    time_evolution_file << "\"\n";
    time_evolution_file.flush();
}

void Wolff::simulate_phase_transition(bool save_time_evolution) {
    const int equilibration_steps = 10000 + 10*N;
    const int decorrelation_steps = N;
    const int snapshot_interval = N;
    
    std::cout << "\nIniciando simulación Ising 2D (Algoritmo Wolff)\n";
    std::cout << "============================================\n";
    std::cout << "Tamaño de red: " << L << "x" << L << " (" << N << " spines)\n";
    std::cout << "Rango de temperatura: [" << T_MIN << "," << T_MAX << "] paso " << T_STEP << "\n";
    std::cout << "Iteraciones por temperatura: " << IT << "\n";
    std::cout << "Guardar evolución temporal: " << (save_time_evolution ? "SI" : "NO") << "\n";
    std::cout << "============================================\n\n";

    int temp_steps = static_cast<int>((T_MAX - T_MIN)/T_STEP) + 1;
    
    for(int t_step = 0; t_step < temp_steps; ++t_step) {
        float T = T_MIN + t_step * T_STEP;
        auto start_time = std::chrono::steady_clock::now();
        
        // Reiniciar archivo CSV para cada temperatura
        if (time_evolution_file.is_open()) {
            time_evolution_file.close();
        }

        // Fase de equilibración
        for(int i = 0; i < equilibration_steps; ++i) {
            update(lattice.get_lattice(), T, lattice.J);
        }

        // Mediciones
        float sum_M = 0.0f, sum_M2 = 0.0f, sum_M4 = 0.0f;
        float sum_E = 0.0f, sum_E2 = 0.0f;
        int samples = 0;
        int time_step = 0;

        for(long i = 0; i < IT; ++i) {
            update(lattice.get_lattice(), T, lattice.J);
            
            if(save_time_evolution && (i % snapshot_interval == 0)) {
                save_time_step_data(lattice.get_lattice(), T, time_step);
                time_step++;
            }
            
            if(i % decorrelation_steps == 0) {
                float M = calculate_magnetization_per_site(lattice.get_lattice());
                float E = calculate_energy_per_site(lattice.get_lattice());
                
                sum_M += std::abs(M);
                sum_M2 += M * M;
                sum_M4 += M * M * M * M;
                sum_E += E;
                sum_E2 += E * E;
                ++samples;
            }
        }

        // Cálculo de promedios
        float avg_M = sum_M / samples;
        float avg_M2 = sum_M2 / samples;
        float avg_M4 = sum_M4 / samples;
        float avg_E = sum_E / samples;
        float avg_E2 = sum_E2 / samples;
        float var_M = avg_M2 - avg_M * avg_M;
        float var_E = avg_E2 - avg_E * avg_E;

        // Almacenar resultados
        Temperatures.push_back(T);
        MagnetizationResults.push_back(avg_M);
        EnergyResults.push_back(avg_E);
        Susceptibility.push_back((N/T) * var_M);
        SpecificHeat.push_back(var_E/(T*T*N));
        BinderCumulant.push_back(1.0f - avg_M4/(3.0f*avg_M2*avg_M2));

        // Mostrar progreso
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        std::cout << "T = " << std::setw(5) << T 
                  << "  |M| = " << std::setw(8) << avg_M 
                  << "  E = " << std::setw(8) << avg_E
                  << "  χ = " << std::setw(8) << Susceptibility.back()
                  << "  C = " << std::setw(8) << SpecificHeat.back()
                  << "  U = " << std::setw(8) << BinderCumulant.back()
                  << "  [" << elapsed << "s]\n";
        
        if(save_time_evolution && time_evolution_file.is_open()) {
            time_evolution_file.close();
        }
    }

    std::cout << "\nSimulación completada exitosamente!\n";
}

float Wolff::calculate_magnetization_per_site(const std::vector<int>& lattice) const {
    return std::accumulate(lattice.begin(), lattice.end(), 0.0f) / N;
}

float Wolff::calculate_energy_per_site(const std::vector<int>& lattice) const {
    float energy = 0.0f;
    for(int i = 0; i < N; ++i) {
        int row = i / L;
        int col = i % L;
        energy -= lattice[i] * (
            lattice[row * L + (col + 1) % L] +  // Vecino derecho
            lattice[((row + 1) % L) * L + col]  // Vecino inferior
        );
    }
    return energy / N;
}

void Wolff::add_to_cluster(std::vector<int>& lattice, std::unordered_set<int>& cluster, 
                         std::queue<int>& spin_queue, float P, int i) {
    if (cluster.find(i) == cluster.end()) {
        cluster.insert(i);
        int row = i / L;
        int col = i % L;
        std::vector<int> neighbors = {
            ((row + 1) % L) * L + col,  // Abajo
            ((row - 1 + L) % L) * L + col,  // Arriba
            row * L + (col + 1) % L,  // Derecha
            row * L + (col - 1 + L) % L   // Izquierda
        };

        for (int neighbor : neighbors) {
            if (lattice[neighbor] == lattice[i] && dis(gen) < P) {
                spin_queue.push(neighbor);
            }
        }
    }
}

void Wolff::update(std::vector<int>& lattice, float T, float J) {
    float P = 1.0f - expf(-2.0f * J / T);
    std::uniform_int_distribution<> dist(0, N-1);
    int seed = dist(gen);
    
    std::unordered_set<int> cluster;
    std::queue<int> spin_queue;
    spin_queue.push(seed);
    
    while (!spin_queue.empty()) {
        int current = spin_queue.front();
        spin_queue.pop();
        add_to_cluster(lattice, cluster, spin_queue, P, current);
    }
    
    for (int i : cluster) {
        lattice[i] *= -1;
    }
}

void Wolff::store_results_to_file() const {
    std::string filename = "ising_results_L" + std::to_string(L) + ".csv";
    std::ofstream outFile(filename);
    
    if (!outFile) {
        throw std::runtime_error("No se pudo abrir el archivo de resultados");
    }
    
    outFile << "Temperature,Magnetization,Energy,Susceptibility,SpecificHeat,BinderCumulant\n";
    outFile << std::scientific << std::setprecision(8);
    
    for(size_t i = 0; i < Temperatures.size(); ++i) {
        outFile << Temperatures[i] << ","
                << MagnetizationResults[i] << ","
                << EnergyResults[i] << ","
                << Susceptibility[i] << ","
                << SpecificHeat[i] << ","
                << BinderCumulant[i] << "\n";
    }
    
    std::cout << "\nResultados guardados en: " << filename << "\n";
}
