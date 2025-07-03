import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import savgol_filter
import warnings
import os
import argparse
import re
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator
import pandas as pd

# =============================================
# CONFIGURACIÓN GLOBAL Y CONSTANTES
# =============================================

# Paleta de colores profesional (definida al inicio)
COLORS = {
    'magnetization': '#1f77b4',
    'susceptibility': '#ff7f0e',
    'specific_heat': '#2ca02c',
    'binder': '#9467bd',
    'energy': '#d62728',
    'critical': '#7f7f7f',
    'background': '#f5f5f5'
}

# =============================================
# CONFIGURACIÓN AVANZADA DE ESTILO
# =============================================

def configure_style():
    """Configuración profesional de estilo para gráficas científicas"""
    available_styles = plt.style.available
    preferred_order = [
        'seaborn-v0_8-paper',
        'seaborn-paper', 
        'seaborn',
        'ggplot',
        'classic'
    ]
    
    selected_style = 'default'
    for style in preferred_order:
        if style in available_styles:
            selected_style = style
            break
    
    plt.style.use(selected_style)
    
    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.figsize': (8, 10),
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'axes.labelpad': 4.0,
        'figure.autolayout': True,
        'mathtext.fontset': 'stix',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '0.15',
        'axes.labelcolor': '0.15',
        'xtick.color': '0.15',
        'ytick.color': '0.15',
    })
    
    return selected_style

# =============================================
# FUNCIONES DE CARGA DE DATOS
# =============================================

def load_data(filename):
    """Carga los datos del archivo CSV de manera robusta"""
    try:
        # Primero intentamos con pandas que maneja mejor los encabezados
        try:
            df = pd.read_csv(filename, comment='#')
            
            # Verificamos las columnas necesarias
            required_columns = ['Temperature', 'Magnetization', 'Energy', 
                              'Susceptibility', 'SpecificHeat', 'BinderCumulant']
            
            # Si el archivo tiene encabezados diferentes, intentamos mapear
            if not all(col in df.columns for col in required_columns):
                if len(df.columns) == 6:
                    df.columns = required_columns
                else:
                    raise ValueError("Número de columnas incorrecto")
            
            return {
                'T': df['Temperature'].values,
                'M': df['Magnetization'].values,
                'E': df['Energy'].values,
                'chi': df['Susceptibility'].values,
                'C': df['SpecificHeat'].values,
                'U': df['BinderCumulant'].values
            }
            
        except Exception as pd_err:
            # Si falla pandas, intentamos con numpy
            data = np.loadtxt(filename, comments='#', delimiter=',')
            if data.ndim != 2 or data.shape[1] != 6:
                raise ValueError("El archivo debe contener exactamente 6 columnas de datos")
            
            return {
                'T': data[:,0],
                'M': data[:,1],
                'E': data[:,2],
                'chi': data[:,3],
                'C': data[:,4],
                'U': data[:,5]
            }
            
    except Exception as e:
        raise ValueError(f"Error al cargar datos: {str(e)}\nFormato esperado:\n" +
                        "Temperature,Magnetization,Energy,Susceptibility,SpecificHeat,BinderCumulant\n" +
                        "O 6 columnas numéricas sin encabezado")

def extract_parameters(filename):
    """Extrae parámetros con expresiones regulares robustas"""
    defaults = {'L': 32, 'J': 1.0, 'T_min': 1.0, 'T_max': 3.0, 'IT': 1000}
    
    try:
        with open(filename, 'r') as f:
            content = f.read(1000)  # Leer solo los primeros 1000 caracteres
            
        params = {
            'L': int(re.search(r'L\s*[=:]\s*(\d+)', content).group(1)) 
                 if re.search(r'L\s*[=:]\s*\d+', content) else defaults['L'],
            'J': float(re.search(r'J\s*[=:]\s*([\d\.]+)', content).group(1)) 
                 if re.search(r'J\s*[=:]\s*[\d\.]+', content) else defaults['J'],
            'T_min': float(re.search(r'T\s*min\s*[=:]\s*([\d\.]+)', content, re.IGNORECASE).group(1)) 
                 if re.search(r'T\s*min\s*[=:]\s*[\d\.]+', content, re.IGNORECASE) else defaults['T_min'],
            'T_max': float(re.search(r'T\s*max\s*[=:]\s*([\d\.]+)', content, re.IGNORECASE).group(1)) 
                 if re.search(r'T\s*max\s*[=:]\s*[\d\.]+', content, re.IGNORECASE) else defaults['T_max'],
            'IT': int(re.search(r'Iterations?\s*[=:]\s*(\d+)', content, re.IGNORECASE).group(1)) 
                 if re.search(r'Iterations?\s*[=:]\s*\d+', content, re.IGNORECASE) else defaults['IT']
        }
        
        return params
    
    except Exception as e:
        warnings.warn(f"Error leyendo parámetros: {str(e)}. Usando valores por defecto.")
        return defaults

# =============================================
# FUNCIONES DE VISUALIZACIÓN
# =============================================

def plot_phase_transition(data, params, output_file):
    """Visualización completa de la transición de fase"""
    fig = plt.figure(figsize=(8, 10))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.4)
    
    # Configurar ejes
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    
    # Suavizado adaptativo
    # Versión corregida del código
 # Versión corregida - asegúrate de que la indentación sea consistente
    window_size = max(3, min(11, len(data['T'])//5))  # Esta línea sin indentación
    if window_size % 2 == 0:  # Esta línea al mismo nivel
      window_size += 1  # Esta línea indentada con 4 espacios
    
    # 1. Magnetización
    ax1.plot(data['T'], data['M'], 'o', color=COLORS['magnetization'], 
            markersize=5, alpha=0.7, label='Datos')
    if len(data['T']) > window_size:
        M_smooth = savgol_filter(data['M'], window_size, 2)
        ax1.plot(data['T'], M_smooth, '-', color=COLORS['magnetization'], 
               linewidth=2, label='Suavizado')
    ax1.set_ylabel(r'Magnetización $\langle|M|\rangle$')
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # 2. Susceptibilidad
    ax2.plot(data['T'], data['chi'], 'o', color=COLORS['susceptibility'], 
            markersize=5, alpha=0.7, label='Datos')
    if len(data['T']) > window_size:
        chi_smooth = savgol_filter(data['chi'], window_size, 2)
        ax2.plot(data['T'], chi_smooth, '-', color=COLORS['susceptibility'], 
               linewidth=2, label='Suavizado')
    ax2.set_ylabel(r'Susceptibilidad $\chi$')
    ax2.legend(loc='upper right', framealpha=0.9)
    
    # 3. Calor específico
    ax3.plot(data['T'], data['C'], 'o', color=COLORS['specific_heat'], 
            markersize=5, alpha=0.7, label='Datos')
    if len(data['T']) > window_size:
        C_smooth = savgol_filter(data['C'], window_size, 2)
        ax3.plot(data['T'], C_smooth, '-', color=COLORS['specific_heat'], 
               linewidth=2, label='Suavizado')
    ax3.set_ylabel(r'Calor Específico $C$')
    ax3.legend(loc='upper right', framealpha=0.9)
    
    # 4. Cumulante de Binder
    ax4.plot(data['T'], data['U'], 'o', color=COLORS['binder'], 
            markersize=5, alpha=0.7, label='Datos')
    if len(data['T']) > window_size:
        U_smooth = savgol_filter(data['U'], window_size, 2)
        ax4.plot(data['T'], U_smooth, '-', color=COLORS['binder'], 
               linewidth=2, label='Suavizado')
    ax4.set_xlabel('Temperatura $T$')
    ax4.set_ylabel(r'Cumulante de Binder $U$')
    ax4.legend(loc='upper right', framealpha=0.9)
    
    # Añadir línea crítica teórica (2.269 para Ising 2D)
    Tc = 2.269
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(x=Tc, color=COLORS['critical'], linestyle='--', 
                  linewidth=1, alpha=0.7, zorder=0)
        ax.text(Tc+0.02, ax.get_ylim()[1]*0.9, r'$T_c \approx 2.269$', 
               color=COLORS['critical'], fontsize=9, alpha=0.9)
        
        # Configuración adicional de ejes
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which='major', alpha=0.4)
        ax.grid(which='minor', alpha=0.1)
    
    # Título y guardado
    fig.suptitle(
        f"Transición de Fase - Modelo de Ising 2D\n"
        f"L = {params['L']}, J = {params['J']:.2f}, "
        f"T ∈ [{params['T_min']:.2f}, {params['T_max']:.2f}], "
        f"{params['IT']:,} iteraciones",
        y=0.98
    )
    
    plt.savefig(output_file)
    plt.close()

def plot_energy_analysis(data, params, output_file):
    """Visualización detallada de la energía"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Suavizado adaptativo
    window_size = max(3, min(11, len(data['T']) // 5))
    if window_size % 2 == 0: window_size += 1
    
    # 1. Energía vs Temperatura
    ax1.plot(data['T'], data['E'], 'o', color=COLORS['energy'], 
            markersize=5, alpha=0.7, label='Datos')
    if len(data['T']) > window_size:
        E_smooth = savgol_filter(data['E'], window_size, 2)
        ax1.plot(data['T'], E_smooth, '-', color=COLORS['energy'], 
               linewidth=2, label='Suavizado')
    ax1.set_ylabel('Energía por sitio $E$')
    ax1.legend(loc='best', framealpha=0.9)
    
    # 2. Derivada numérica de la energía
    if len(data['T']) > 1:
        dE = np.gradient(data['E'], data['T'])
        ax2.plot(data['T'], dE, 'o', color='#8c564b', 
                markersize=5, alpha=0.7, label='Derivada numérica')
        if len(data['T']) > window_size:
            dE_smooth = savgol_filter(dE, window_size, 2)
            ax2.plot(data['T'], dE_smooth, '-', color='#8c564b', 
                   linewidth=2, label='Suavizado')
    ax2.set_xlabel('Temperatura $T$')
    ax2.set_ylabel(r'$\partial E/\partial T$')
    ax2.legend(loc='best', framealpha=0.9)
    
    # Línea crítica
    Tc = 2.269
    for ax in [ax1, ax2]:
        ax.axvline(x=Tc, color=COLORS['critical'], linestyle='--', 
                  linewidth=1, alpha=0.7, zorder=0)
        ax.text(Tc+0.02, ax.get_ylim()[1]*0.9, r'$T_c \approx 2.269$', 
               color=COLORS['critical'], fontsize=9, alpha=0.9)
        
        # Configuración de ejes
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which='major', alpha=0.4)
        ax.grid(which='minor', alpha=0.1)
    
    # Título y guardado
    fig.suptitle(
        f"Análisis de Energía - Modelo de Ising 2D\n"
        f"L = {params['L']}, J = {params['J']:.2f}",
        y=0.98
    )
    
    plt.savefig(output_file)
    plt.close()

# =============================================
# EJECUCIÓN PRINCIPAL
# =============================================

def main():
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(
        description='Visualización profesional de resultados de simulación del modelo de Ising 2D',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_file', help='Archivo de datos de entrada')
    parser.add_argument('-o', '--output', default='ising_results', 
                       help='Prefijo para archivos de salida')
    parser.add_argument('--L', type=int, help='Tamaño de la red L×L')
    parser.add_argument('--J', type=float, help='Constante de acoplamiento J')
    parser.add_argument('--Tmin', type=float, help='Temperatura mínima')
    parser.add_argument('--Tmax', type=float, help='Temperatura máxima')
    parser.add_argument('--iter', type=int, help='Número de iteraciones')
    parser.add_argument('--format', default='pdf', choices=['pdf', 'png', 'svg'],
                       help='Formato de salida para gráficos')
    
    args = parser.parse_args()
    
    try:
        # Configuración inicial
        selected_style = configure_style()
        rcParams['savefig.format'] = args.format
        
        print("\n" + "="*70)
        print("VISUALIZACIÓN DE RESULTADOS - MODELO DE ISING 2D".center(70))
        print("="*70)
        
        # Cargar datos y parámetros
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Archivo no encontrado: {args.input_file}")
        
        data = load_data(args.input_file)
        params = extract_parameters(args.input_file)
        
        # Sobrescribir parámetros si se especifican
        if args.L: params['L'] = args.L
        if args.J: params['J'] = args.J
        if args.Tmin: params['T_min'] = args.Tmin
        if args.Tmax: params['T_max'] = args.Tmax
        if args.iter: params['IT'] = args.iter
        
        # Mostrar resumen
        print("\nPARÁMETROS DE SIMULACIÓN:")
        print(f"• Tamaño de red: {params['L']}×{params['L']}")
        print(f"• Constante de acoplamiento (J): {params['J']:.4f}")
        print(f"• Rango de temperatura: [{params['T_min']:.4f}, {params['T_max']:.4f}]")
        print(f"• Iteraciones por temperatura: {params['IT']:,}")
        print(f"• Puntos de datos: {len(data['T'])}")
        print(f"• Estilo gráfico seleccionado: {selected_style}")
        
        # Generar gráficos
        print("\nGENERANDO GRÁFICOS...")
        phase_file = f"{args.output}_phase_transition.{args.format}"
        energy_file = f"{args.output}_energy_analysis.{args.format}"
        
        plot_phase_transition(data, params, phase_file)
        plot_energy_analysis(data, params, energy_file)
        
        print("\nRESULTADOS GUARDADOS:")
        print(f"• Gráfico de transición de fase: {os.path.abspath(phase_file)}")
        print(f"• Análisis de energía: {os.path.abspath(energy_file)}")
        print("\n" + "="*70)
        print("PROCESO COMPLETADO EXITOSAMENTE".center(70))
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {str(e)}")
        print("Verifique la ruta al archivo de datos.")
    except ValueError as e:
        print(f"\nERROR EN DATOS: {str(e)}")
    except Exception as e:
        print(f"\nERROR INESPERADO: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')

if __name__ == "__main__":
    main()
