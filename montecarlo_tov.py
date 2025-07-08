import numpy as np
import matplotlib.pyplot as plt

def simulacion_tov(n_sim=1000000):
    # Valor nominal del TOV y nivel del líquido
    TOV_nominal = 435.73  # Bbl
    nivel_liquido_mm = 2383  # mm

    # ----------------------------- #
    # 1. Calibración del equipo (NORMAL)
    uc1 = 0.33 / 2.01  # = 0.1642 mm
    error1 = np.random.normal(0, uc1, n_sim)

    # 2. Resolución del equipo (RECTANGULAR)
    uc2 = 1   # = 0.5774 mm
    error2 = np.random.uniform(-uc2, uc2, n_sim)

    # 3. Lectura del observador (RECTANGULAR)
    uc3 = 1   # = 0.5774 mm
    error3 = np.random.uniform(-uc3, uc3, n_sim)

    # 4. Repetibilidad (NORMAL)
    uc4 = 3.3333  # mm
    error4 = np.random.normal(0, uc4, n_sim)

    # 5. Efecto temperatura (RECTANGULAR)
    alfa = 0.0000062  # 1/°F
    deltaT = 91.4 - 68.99  # °F
    uc5 = (alfa * deltaT * nivel_liquido_mm * 10)  # ≈ 1.9116 mm
    error5 = np.random.uniform(-uc5, uc5, n_sim)
    print(uc5)
    # 6. Movimiento del plato (RECTANGULAR)
    uc6 = 0 / np.sqrt(3)  # = 0
    error6 = np.random.uniform(-uc6, uc6, n_sim)

    # 7. Calibración tabla de aforo (NORMAL)
    uc7 = 1.22 / 2  # = 0.6100 Bbl
    error7 = np.random.normal(0, uc7, n_sim)

    # 8. Resolución tabla de aforo (RECTANGULAR)
    uc8 = (TOV_nominal / nivel_liquido_mm) * 1  # Bbl/mm * mm = Bbl
    uc8 /= np.sqrt(3)
    error8 = np.random.uniform(-uc8, uc8, n_sim)

    # ----------------------------- #
    # Errores que afectan el nivel (en mm)
    error_total_nivel_mm = error1 + error2 + error3 + error4 + error5 + error6

    # Convertir a volumen (Bbl) usando coeficiente de sensibilidad
    coef_sens_mm = TOV_nominal / nivel_liquido_mm  # Bbl/mm
    error_vol_mm = error_total_nivel_mm * coef_sens_mm

    # Errores que afectan directamente el volumen (Bbl)
    error_vol_bbl = error7 + error8

    # Sumar todos los errores en volumen
    error_total_vol = error_vol_mm + error_vol_bbl

    # Simulación del TOV
    tov_simulado = TOV_nominal + error_total_vol

    # --- Análisis de Monte Carlo puro --- #
    media_tov = np.mean(tov_simulado)
    uc = np.std(tov_simulado, ddof=1)

    # Percentiles para el intervalo del 95%
    li = np.percentile(tov_simulado, 2.5)
    ls = np.percentile(tov_simulado, 97.5)
    intervalo_95 = ls - li

    # Factor de cobertura estimado por Monte Carlo
    k_mc = (intervalo_95 / 2) / uc
    u_exp_mc = uc * k_mc

    # Mostrar resultados
    print("🔷 Simulación Monte Carlo para TOV")
    print(f"🔹 Incertidumbre típica combinada (uc): {uc:.4f} Bbl")
    print(f"🔹 Incertidumbre expandida (U = uc · k): {intervalo_95/2:.4f} Bbl\n")
    print(f"🔹 Factor de cobertura: {k_mc:.4f}")

    return tov_simulado, u_exp_mc, k_mc

def graficar_histograma(tov_simulado):
    plt.hist(tov_simulado, bins=100, density=True, color="skyblue", edgecolor="black")
    plt.title("Distribución simulada del TOV (Monte Carlo)")
    plt.xlabel("TOV [Bbl]")
    plt.ylabel("Densidad")
    plt.grid(True)
    plt.axvline(np.mean(tov_simulado), color="red", linestyle="--", label=f"Media: {np.mean(tov_simulado):.2f} Bbl")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tov_simulado, u_exp_mc, k_mc = simulacion_tov()
    graficar_histograma(tov_simulado)
