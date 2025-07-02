#include "wolf.h"
#include <fstream>
#include <cmath>
#include <random>
#include <unordered_set>
#include <queue>
#include <vector>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <numeric>
#include <iostream>

Wolff::Wolff(float interactionStrength, int latticeSize, float T_MIN, float T_MAX, float T_STEP, long int IT):
    lattice(interactionStrength, latticeSize),
    T_MIN(T_MIN), T_MAX(T_MAX), T_STEP(T_STEP),
    L(latticeSize), N(latticeSize * latticeSize), IT(IT),
    gen(std::random_device{}()), dis(0.0, 1.0)
{
    // Reservar espacio para resultados
    Temperatures.reserve((T_MAX-T_MIN)/T_STEP + 1);
    MagnetizationResults.reserve((T_MAX-T_MIN)/T_STEP + 1);
    EnergyResults.reserve((T_MAX-T_MIN)/T_STEP + 1);
    Susceptibility.reserve((T_MAX-T_MIN)/T_STEP + 1);
    SpecificHeat.reserve((T_MAX-T_MIN)/T_STEP + 1);
    BinderCumulant.reserve((T_MAX-T_MIN)/T_STEP + 1);
}

void Wolff::simulate_phase_transition() {
    const int equilibration_steps = 10000;  // Pasos iniciales de equilibración
    const int decorrelation_steps = N;      // Pasos entre mediciones (1 pasada completa por la red)
    
    std::cout << "Iniciando simulación para L=" << L << "\n";
    std::cout << "Configuración:\n";
    std::cout << " - Iteraciones por temperatura: " << IT << "\n";
    std::cout << " - Pasos de decorrelación: " << decorrelation_steps << "\n";
    std::cout << " - Muestras efectivas por temperatura: " << IT/decorrelation_steps << "\n\n";

    for(float T = T_MIN; T <= T_MAX; T += T_STEP) {
        auto start_time = std::chrono::steady_clock::now();
        
        // 1. Equilibración del sistema
        for(int i = 0; i < equilibration_steps; i++) {
            update(lattice.get_lattice(), T, lattice.J);
        }

        // 2. Mediciones con pasos de decorrelación
        double sum_M = 0, sum_M2 = 0, sum_M4 = 0;
        double sum_E = 0, sum_E2 = 0;
        int samples = 0;

        for(long i = 0; i < IT; i++) {
            update(lattice.get_lattice(), T, lattice.J);
            
            if(i % decorrelation_steps == 0) {
                float M = calculate_magnetization_per_site(lattice.get_lattice());
                float E = calculate_energy_per_site(lattice.get_lattice());
                
                sum_M += fabs(M);       // Para magnetización
                sum_M2 += M * M;        // Para susceptibilidad
                sum_M4 += M * M * M * M; // Para cumulante de Binder
                sum_E += E;             // Para energía
                sum_E2 += E * E;        // Para calor específico
                samples++;
            }
        }

        // 3. Cálculo de promedios
        float avg_M = sum_M / samples;
        float avg_M2 = sum_M2 / samples;
        float avg_M4 = sum_M4 / samples;
        float avg_E = sum_E / samples;
        float avg_E2 = sum_E2 / samples;

        // 4. Almacenar resultados
        Temperatures.push_back(T);
        MagnetizationResults.push_back(avg_M);
        EnergyResults.push_back(avg_E);
        
        // Susceptibilidad usando M real (no |M|)
        Susceptibility.push_back((N/T) * (avg_M2 - avg_M*avg_M));
        
        // Calor específico
        SpecificHeat.push_back((avg_E2 - avg_E*avg_E)/(T*T));
        
        // Cumulante de Binder usando M real
        BinderCumulant.push_back(1 - avg_M4/(3*avg_M2*avg_M2));

        // Mostrar progreso
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        std::cout << "T=" << T << " completado en " << elapsed << "s. "
                  << "M=" << avg_M << " χ=" << Susceptibility.back() 
                  << " U=" << BinderCumulant.back() << "\n";
    }

    std::cout << "\nSimulación completada. Resultados guardados.\n";
}

float Wolff::calculate_magnetization_per_site(const std::vector<int>& lattice) {
    int total_spin = std::accumulate(lattice.begin(), lattice.end(), 0);
    return static_cast<float>(total_spin) / N;
}

float Wolff::calculate_energy_per_site(const std::vector<int>& lattice) {
    float energy = 0.0;
    for(int i = 0; i < N; ++i) {
        // Interacción con vecinos derecho e inferior (evita contar dos veces)
        energy -= lattice[i] * (lattice[(i+1)%L + (i/L)*L] + lattice[(i+L)%N]);
    }
    return energy / N;
}

void Wolff::add_to_cluster(std::vector<int>& lattice, std::unordered_set<int>& cluster, 
                          std::queue<int>& spin_queue, float P, int i) {
    if (cluster.find(i) == cluster.end()) {
        cluster.insert(i);
        std::vector<int> neighbors = {
            (i + L) % N,          // Abajo
            (i - L + N) % N,      // Arriba 
            (i + 1) % L + (i / L) * L,   // Derecha
            (i - 1 + L) % L + (i / L) * L // Izquierda
        };

        for (int neighbor : neighbors) {
            if (lattice[neighbor] == lattice[i] && dis(gen) < P) {
                spin_queue.push(neighbor);
            }
        }
    }
}

void Wolff::update(std::vector<int>& lattice, float T, float J) {
    float P = 1 - exp(-2 * J / T);
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
    
    // Voltear el cluster
    for (int i : cluster) {
        lattice[i] *= -1;
    }
}

void Wolff::store_results_to_file() const {
    std::string filename = "results_L" + std::to_string(L) + ".dat";
    std::ofstream outFile(filename);
    
    if (!outFile) {
        throw std::runtime_error("No se pudo abrir el archivo para escritura: " + filename);
    }
    
    // Encabezado con parámetros de simulación
    outFile << "# Modelo de Ising 2D - Algoritmo Wolff\n";
    outFile << "# L = " << L << ", J = " << lattice.J << ", Iteraciones = " << IT << "\n";
    outFile << "Temperatura Magnetizacion Energia Susceptibilidad CalorEspecifico CumulanteBinder\n";
    
    // Formato científico con precisión adecuada
    outFile << std::scientific << std::setprecision(6);
    
    for(size_t i = 0; i < Temperatures.size(); ++i) {
        outFile << Temperatures[i] << " "
                << MagnetizationResults[i] << " "
                << EnergyResults[i] << " "
                << Susceptibility[i] << " "
                << SpecificHeat[i] << " "
                << BinderCumulant[i] << "\n";
    }
    
    std::cout << "Resultados guardados en: " << filename << "\n";
}
