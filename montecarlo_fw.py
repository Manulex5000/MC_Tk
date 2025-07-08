import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def simulacion_fw(n_sim=1000000):
    # Valor nominal
    FW = 0.0  # Bbl
    nivel_mm = 2383

    # --- Fuentes de incertidumbre y coeficientes de sensibilidad ---
    # 1. Calibración de la cinta (NORMAL)
    u1 = 0.33 / 2.01
    s1 = 0.2

    # 2. Resolución de la cinta (RECTANGULAR)
    u2 = 1 / np.sqrt(3)
    s2 = 0.2

    # 3. Apreciación del observador (RECTANGULAR)
    u3 = 1 / np.sqrt(3)
    s3 = 0.2

    # 4. Repetibilidad (NORMAL, ajustada con desviación de 3 datos)
    u4 = (0.577 / np.sqrt(3)) * 10  # ≈ 3.333
    s4 = 0.2

    # 5. Efecto de la temperatura (RECTANGULAR)
    alfa = 0.0000062  # coef. de expansión térmica
    deltaT = 91.40 - 68.99
    u5 = (alfa * deltaT * nivel_mm) / np.sqrt(3)  # ≈ 3.3110 mm
    s5 = 0.2

    # 6. Movimiento del plato (RECTANGULAR)
    u6 = 0 / np.sqrt(3)
    s6 = 0.2

    # 7. Calibración de la tabla de aforo (NORMAL)
    u7 = 0 / 2  # ya que FW = 0
    s7 = 1.0

    # 8. Resolución de la tabla de aforo (RECTANGULAR)
    u8 = 0 / np.sqrt(3)  # ya que FW = 0
    s8 = 1.0

    # --- Simulaciones Monte Carlo para cada fuente ---
    e1 = np.random.normal(0, u1, n_sim)
    e2 = np.random.uniform(-u2, u2, n_sim)
    e3 = np.random.uniform(-u3, u3, n_sim)
    e4 = np.random.normal(0, u4, n_sim)
    e5 = np.random.uniform(-u5, u5, n_sim)
    e6 = np.random.uniform(-u6, u6, n_sim)
    e7 = np.random.normal(0, u7, n_sim)
    e8 = np.random.uniform(-u8, u8, n_sim)

    # --- Aplicar coeficientes de sensibilidad ---
    error_nivel = (
        s1 * e1 +
        s2 * e2 +
        s3 * e3 +
        s4 * e4 +
        s5 * e5 +
        s6 * e6
    )
    error_tabla = s7 * e7 + s8 * e8

    # --- Suma total de errores en volumen ---
    fw_simulado = FW + error_nivel + error_tabla

    # --- Estadísticos ---
    media = np.mean(fw_simulado)
    uc = np.std(fw_simulado, ddof=1)
    intervalo_95 = norm.interval(0.95, loc=media, scale=uc)
    k = (intervalo_95[1] - intervalo_95[0]) / (2 * uc)
    u_exp_mc = uc * k

    print("🔷 Simulación Monte Carlo para FW")
    print(f"🔹 Incertidumbre típica combinada (uc): {uc:.6f} Bbl")
    print(f"🔹 Incertidumbre expandida (U = uc · k): {u_exp_mc:.6f} Bbl")
    print(f"🔹 Factor de cobertura efectivo (k): {k:.6f}")
    return fw_simulado, u_exp_mc, k


def graficar_histograma(fw_simulado):
    plt.hist(fw_simulado, bins=100, density=True, color="lightgreen", edgecolor="black")
    plt.title("Distribución simulada del FW (Monte Carlo)")
    plt.xlabel("FW [Bbl]")
    plt.ylabel("Densidad")
    plt.grid(True)
    plt.axvline(np.mean(fw_simulado), color="red", linestyle="--", label=f"Media: {np.mean(fw_simulado):.6f} Bbl")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Ejecutar simulación y graficar solo si se corre directamente
if __name__ == "__main__":
    fw_simulado, u_exp_mc, k = simulacion_fw()
    graficar_histograma(fw_simulado)