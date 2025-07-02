#include "wolf.h"
#include <iostream>
#include <limits>
#include <cmath>

template<typename T>
T read_input(const std::string& prompt, T min = std::numeric_limits<T>::lowest(), 
             T max = std::numeric_limits<T>::max()) {
    T value;
    while (true) {
        std::cout << prompt;
        if (!(std::cin >> value)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cerr << "Error: Entrada inválida. Por favor ingrese un número.\n";
            continue;
        }
        if (value < min || value > max) {
            std::cerr << "Error: El valor debe estar entre " << min << " y " << max << ".\n";
            continue;
        }
        break;
    }
    return value;
}

int main() {
    try {
        std::cout << "=============================================\n"
                  << "    Simulación del Modelo de Ising con Wolff\n"
                  << "=============================================\n\n";

        int L = read_input<int>("Tamaño de red L (potencia de 2, >=2): ", 2);
        float T_MIN = read_input<float>("Temperatura mínima T_MIN (>0): ", 0.01f);
        float T_MAX = read_input<float>("Temperatura máxima T_MAX (>T_MIN): ", T_MIN + 0.01f);
        float T_STEP = read_input<float>("Paso de temperatura T_STEP (>0): ", 0.001f, (T_MAX - T_MIN)/2);
        float J = read_input<float>("Constante de acoplamiento J: ");
        long IT = read_input<long>("Iteraciones IT (recomendado ≥ 10000): ", 1000);

        Wolff simulation(J, L, T_MIN, T_MAX, T_STEP, IT);
        simulation.simulate_phase_transition();
        simulation.store_results_to_file();

        std::cout << "\nSimulación completada. Resultados guardados en results_L" << L << ".dat\n";

    } catch (const std::exception& e) {
        std::cerr << "\nError fatal: " << e.what() << "\n";
        return 1;
    }

    return 0;
}