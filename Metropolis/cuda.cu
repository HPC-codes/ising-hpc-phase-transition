#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <fstream>

// Parámetros ajustados para mejor precisión
#define L 32
#define N (L*L)
#define J 1.0f
#define THERMALIZATION 100000  // Pasos de termalización
#define ITERATIONS 1000000     // Iteraciones de medición
#define NTHREADS 256
#define TEMP_START 1.0f
#define TEMP_END 3.0f
#define TEMP_STEP 0.1f
#define MEASUREMENT_INTERVAL 100

__global__ void setup_rand_kernel(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

__global__ void initialize_lattice_kernel(int* lattice, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        lattice[idx] = (curand_uniform(&states[idx]) < 0.5f) ? -1 : 1;
    }
}

__device__ int get_index(int row, int col) {
    return (row * L + col);
}

__device__ float calculate_energy_change(int* lattice, int idx, float T) {
    int row = idx / L;
    int col = idx % L;
    
    int up = get_index((row - 1 + L) % L, col);
    int down = get_index((row + 1) % L, col);
    int left = get_index(row, (col - 1 + L) % L);
    int right = get_index(row, (col + 1) % L);
    
    int spin = lattice[idx];
    int sum_neighbors = lattice[up] + lattice[down] + lattice[left] + lattice[right];
    return 2.0f * J * spin * sum_neighbors;
}

__global__ void metropolis_sweep(int* lattice, float* total_energy, curandState* states, float T, bool even_sweep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int row = idx / L;
    int col = idx % L;
    
    // Actualización checkerboard
    if ((row + col) % 2 == even_sweep) {
        float delta_E = calculate_energy_change(lattice, idx, T);
        float rnd = curand_uniform(&states[idx]);
        
        if (delta_E <= 0.0f || rnd < __expf(-delta_E / T)) {
            lattice[idx] = -lattice[idx];
            atomicAdd(total_energy, 2.0f * delta_E);
        }
    }
}

__global__ void calculate_observables(int* lattice, float* magnetization, float* energy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int spin = lattice[idx];
        atomicAdd(magnetization, spin);
        
        // Calcular energía solo para una dirección para evitar doble conteo
        if (idx % L != L - 1) {  // Vecino derecho
            atomicAdd(energy, -J * spin * lattice[idx + 1]);
        }
        if (idx / L != L - 1) {  // Vecino inferior
            atomicAdd(energy, -J * spin * lattice[idx + L]);
        }
    }
}

int main() {
    // Configuración CUDA
    dim3 blocks((N + NTHREADS - 1) / NTHREADS);
    dim3 threads(NTHREADS);
    
    // Asignación de memoria
    int* d_lattice;
    float* d_magnetization;
    float* d_energy;
    curandState* d_states;
    
    cudaMalloc((void**)&d_lattice, N * sizeof(int));
    cudaMalloc((void**)&d_magnetization, sizeof(float));
    cudaMalloc((void**)&d_energy, sizeof(float));
    cudaMalloc((void**)&d_states, N * sizeof(curandState));
    
    // Inicialización
    setup_rand_kernel<<<blocks, threads>>>(d_states, time(nullptr));
    initialize_lattice_kernel<<<blocks, threads>>>(d_lattice, d_states);
    
    // Archivo de resultados
    std::ofstream results("ising_results.csv");
    results << "Temperature,Magnetization,Energy,Susceptibility,SpecificHeat,BinderCumulant\n";
    results << std::scientific << std::setprecision(6);
    
    // Bucle de temperatura
    for (float T = TEMP_START; T <= TEMP_END; T += TEMP_STEP) {
        clock_t start = clock();
        
        // Termalización
        for (int i = 0; i < THERMALIZATION; i++) {
            float zero = 0.0f;
            cudaMemcpy(d_energy, &zero, sizeof(float), cudaMemcpyHostToDevice);
            
            metropolis_sweep<<<blocks, threads>>>(d_lattice, d_energy, d_states, T, true);
            metropolis_sweep<<<blocks, threads>>>(d_lattice, d_energy, d_states, T, false);
        }
        
        // Mediciones
        float sum_M = 0.0f, sum_M2 = 0.0f, sum_M4 = 0.0f;
        float sum_E = 0.0f, sum_E2 = 0.0f;
        int measurements = 0;
        
        for (int i = 0; i < ITERATIONS; i++) {
            float zero = 0.0f;
            cudaMemcpy(d_energy, &zero, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_magnetization, &zero, sizeof(float), cudaMemcpyHostToDevice);
            
            metropolis_sweep<<<blocks, threads>>>(d_lattice, d_energy, d_states, T, true);
            metropolis_sweep<<<blocks, threads>>>(d_lattice, d_energy, d_states, T, false);
            
            if (i % MEASUREMENT_INTERVAL == 0) {
                calculate_observables<<<blocks, threads>>>(d_lattice, d_magnetization, d_energy);
                cudaDeviceSynchronize();
                
                float M, E;
                cudaMemcpy(&M, d_magnetization, sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&E, d_energy, sizeof(float), cudaMemcpyDeviceToHost);
                
                M /= N;
                E /= N;
                
                sum_M += fabs(M);
                sum_M2 += M * M;
                sum_M4 += M * M * M * M;
                sum_E += E;
                sum_E2 += E * E;
                measurements++;
            }
        }
        
        // Cálculo de promedios
        float avg_M = sum_M / measurements;
        float avg_M2 = sum_M2 / measurements;
        float avg_M4 = sum_M4 / measurements;
        float avg_E = sum_E / measurements;
        float avg_E2 = sum_E2 / measurements;
        
        float var_M = (avg_M2 - avg_M * avg_M) * N;
        float var_E = (avg_E2 - avg_E * avg_E) * N;
        
        float susceptibility = var_M / T;
        float specific_heat = var_E / (T * T);
        float binder_cumulant = 1.0f - avg_M4 / (3.0f * avg_M2 * avg_M2);
        
        // Guardar resultados
        results << T << "," << avg_M << "," << avg_E << "," 
               << susceptibility << "," << specific_heat << "," 
               << binder_cumulant << "\n";
        
        // Mostrar progreso
        clock_t end = clock();
        double elapsed = double(end - start) / CLOCKS_PER_SEC;
        
        std::cout << "T = " << std::setw(4) << T 
                 << "  |M| = " << std::setw(8) << avg_M 
                 << "  E = " << std::setw(8) << avg_E
                 << "  χ = " << std::setw(8) << susceptibility
                 << "  C = " << std::setw(8) << specific_heat
                 << "  U = " << std::setw(8) << binder_cumulant
                 << "  [" << elapsed << "s]\n";
    }
    
    // Liberar memoria
    cudaFree(d_lattice);
    cudaFree(d_magnetization);
    cudaFree(d_energy);
    cudaFree(d_states);
    
    results.close();
    std::cout << "\nResultados guardados en ising_results.csv\n";
    
    return 0;
}