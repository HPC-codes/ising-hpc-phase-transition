#include <omp.h>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <memory>
#include <mutex>
#include <iostream>
#include <chrono>

class SwendsenWangParallelFast {
public:
    SwendsenWangParallelFast(float interactionStrength, int latticeSize, 
                           float T_MIN, float T_MAX, float T_STEP, long int IT)
        : J(interactionStrength), L(latticeSize), N(latticeSize * latticeSize),
          T_MIN(T_MIN), T_MAX(T_MAX), T_STEP(T_STEP), IT(IT) {
        
        if (L <= 0) throw std::invalid_argument("Lattice size must be positive");
        if (T_MIN <= 0.0f || T_MAX <= 0.0f) throw std::invalid_argument("Temperature must be positive");
        if (T_STEP <= 0.0f) throw std::invalid_argument("Temperature step must be positive");
        if (IT <= 0) throw std::invalid_argument("Number of iterations must be positive");

        // Inicializar generadores de números aleatorios para cada hilo
        int max_threads = omp_get_max_threads();
        rngs.resize(max_threads);
        unsigned seed = static_cast<unsigned>(std::time(nullptr));
        for (int i = 0; i < max_threads; ++i) {
            rngs[i].seed(seed + i);
        }
    }

    void simulate_phase_transition() {
        const int equilibration_steps = 100000 + 10*N;
        const int decorrelation_steps = N;
        
        std::cout << "\nIniciando simulación Ising 2D (Algoritmo Swendsen-Wang OpenMP)\n";
        std::cout << "============================================\n";
        std::cout << "Tamaño de red: " << L << "x" << L << " (" << N << " spines)\n";
        std::cout << "Rango de temperatura: [" << T_MIN << "," << T_MAX << "] paso " << T_STEP << "\n";
        std::cout << "Iteraciones por temperatura: " << IT << "\n";
        std::cout << "Número de hilos OpenMP: " << omp_get_max_threads() << "\n";
        std::cout << "============================================\n\n";

        // Pre-reservar espacio para resultados
        int expected_points = static_cast<int>((T_MAX-T_MIN)/T_STEP) + 1;
        Temperatures.reserve(expected_points);
        MagnetizationResults.reserve(expected_points);
        EnergyResults.reserve(expected_points);
        Susceptibility.reserve(expected_points);
        SpecificHeat.reserve(expected_points);
        BinderCumulant.reserve(expected_points);

        // Bucle principal de temperatura
        for (float T = T_MIN; T <= T_MAX; T += T_STEP) {
            auto start_time = std::chrono::steady_clock::now();
            
            // Inicializar red aleatoria
            std::vector<int> lattice(N);
            initialize_lattice(lattice, T);

            // Fase de equilibración
            for (int i = 0; i < equilibration_steps; ++i) {
                swendsen_wang_step(lattice, T);
            }

            // Variables para acumular promedios
            double sum_M = 0.0, sum_M2 = 0.0, sum_M4 = 0.0;
            double sum_E = 0.0, sum_E2 = 0.0;
            int samples = 0;

            // Fase de medición
            for (long i = 0; i < IT; ++i) {
                swendsen_wang_step(lattice, T);
                
                if (i % decorrelation_steps == 0) {
                    float M = calculate_magnetization(lattice);
                    float E = calculate_energy(lattice);
                    
                    sum_M += std::abs(M);
                    sum_M2 += M * M;
                    sum_M4 += M * M * M * M;
                    sum_E += E;
                    sum_E2 += E * E;
                    samples++;
                }
            }

            // Calcular promedios
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
        }

        std::cout << "\nSimulación completada exitosamente!\n";
    }

    void store_results_to_file() const {
        std::string filename = "swendsen_wang_results_L" + std::to_string(L) + ".csv";
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

private:
    float J;
    int L, N;
    float T_MIN, T_MAX, T_STEP;
    long int IT;
    std::vector<std::mt19937> rngs;

    // Resultados
    std::vector<float> Temperatures;
    std::vector<float> MagnetizationResults;
    std::vector<float> EnergyResults;
    std::vector<float> Susceptibility;
    std::vector<float> SpecificHeat;
    std::vector<float> BinderCumulant;

    void initialize_lattice(std::vector<int>& lattice, float T) {
        int thread_id;
        #pragma omp parallel private(thread_id)
        {
            thread_id = omp_get_thread_num();
            std::uniform_real_distribution<float> dist(0.0, 1.0);
            
            #pragma omp for
            for (int i = 0; i < N; ++i) {
                lattice[i] = (dist(rngs[thread_id]) < 0.5 ? -1 : 1);
            }
        }
    }

    float calculate_magnetization(const std::vector<int>& lattice) const {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < N; ++i) {
            sum += lattice[i];
        }
        return sum / N;
    }

    float calculate_energy(const std::vector<int>& lattice) const {
        double energy = 0.0;
        #pragma omp parallel for reduction(+:energy)
        for (int i = 0; i < N; ++i) {
            int row = i / L;
            int col = i % L;
            energy -= lattice[i] * (
                lattice[row * L + (col + 1) % L] +  // Vecino derecho
                lattice[((row + 1) % L) * L + col]  // Vecino inferior
            );
        }
        return energy / N;
    }

    void swendsen_wang_step(std::vector<int>& lattice, float T) {
        std::vector<int> parent(N);
        std::vector<int> rank(N, 0);
        float P = 1.0f - expf(-2.0f * J / T);
        
        // Inicializar estructura union-find
        std::iota(parent.begin(), parent.end(), 0);

        // Paso 1: Formar clusters (paralelizado)
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            std::uniform_real_distribution<float> dist(0.0, 1.0);
            
            #pragma omp for
            for (int i = 0; i < N; ++i) {
                int right = (i % L == L - 1) ? i - (L - 1) : i + 1;
                int down = (i + L) % N;
                
                if (lattice[i] == lattice[right] && dist(rngs[thread_id]) < P) {
                    union_sets(i, right, parent, rank);
                }
                if (lattice[i] == lattice[down] && dist(rngs[thread_id]) < P) {
                    union_sets(i, down, parent, rank);
                }
            }
        }

        // Paso 2: Voltear clusters (paralelizado)
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            std::uniform_int_distribution<int> flip_dist(0, 1);
            
            #pragma omp for
            for (int i = 0; i < N; ++i) {
                if (parent[i] == i) { // Solo procesamos raíces
                    bool flip = flip_dist(rngs[thread_id]);
                    for (int j = 0; j < N; ++j) {
                        if (find_set(j, parent) == i) {
                            lattice[j] = flip ? -lattice[j] : lattice[j];
                        }
                    }
                }
            }
        }
    }

    int find_set(int x, std::vector<int>& parent) {
        while (x != parent[x]) {
            parent[x] = parent[parent[x]]; // Path compression
            x = parent[x];
        }
        return x;
    }

    void union_sets(int x, int y, std::vector<int>& parent, std::vector<int>& rank) {
        x = find_set(x, parent);
        y = find_set(y, parent);
        
        if (x != y) {
            if (rank[x] < rank[y]) {
                std::swap(x, y);
            }
            parent[y] = x;
            if (rank[x] == rank[y]) {
                rank[x]++;
            }
        }
    }
};

int main() {
    // Parámetros idénticos a la simulación CUDA
    float J = 1.0f;
    int L = 128;
    float T_MIN = 1.8f;
    float T_MAX = 2.7f;
    float T_STEP = 0.1f;
    long IT = 1000;

    SwendsenWangParallelFast simulation(J, L, T_MIN, T_MAX, T_STEP, IT);
    simulation.simulate_phase_transition();
    simulation.store_results_to_file();

    return 0;
}