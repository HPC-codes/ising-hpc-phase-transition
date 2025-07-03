#include "wolf.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        // ==============================================
        // CONFIGURACIÓN DE LA SIMULACIÓN
        // ==============================================
        float J = 1.0f;           // Constante de acoplamiento
        int L = 32;               // Tamaño de la red (LxL)
        float T_MIN = 1.5f;       // Temperatura mínima
        float T_MAX = 3.0f;       // Temperatura máxima
        float T_STEP = 0.1f;      // Paso de temperatura
        long IT = 5000;          // Iteraciones por temperatura
        bool save_snapshots = true; // Guardar datos temporales
        
        // ==============================================
        // EJECUCIÓN DE LA SIMULACIÓN
        // ==============================================
        std::cout << "SIMULACION DEL MODELO ISING 2D\n";
        std::cout << "Algoritmo de Wolff - Transición de fase\n";
        std::cout << "======================================\n";
        
        Wolff simulation(J, L, T_MIN, T_MAX, T_STEP, IT);
        simulation.simulate_phase_transition(save_snapshots);
        simulation.store_results_to_file();
        
        std::cout << "\n¡Simulación completada con éxito!\n";
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
