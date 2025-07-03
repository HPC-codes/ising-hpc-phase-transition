#include "wolf.h"
#include <iostream>

int main() {
    try {
        float J = 1.0f;       // Parámetro de interacción
        int L = 32;           // Tamaño de la red
        float T_MIN = 1.8f;   // Temperatura mínima
        float T_MAX = 2.6f;   // Temperatura máxima
        float T_STEP = 0.1f;  // Paso de temperatura
        long IT = 10000;     // Iteraciones por temperatura

        Wolff simulation(J, L, T_MIN, T_MAX, T_STEP, IT);
        simulation.simulate_phase_transition();
        simulation.store_results_to_file();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
