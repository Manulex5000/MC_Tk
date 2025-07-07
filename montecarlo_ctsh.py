import numpy as np
from scipy.stats import norm

def simulacion_ctsh(n_sim=100000):
    # Valores nominales
    Tl_nominal = 91.4   # Temperatura del líquido
    Tamb_nominal = 87.0 # Temperatura ambiente

    # --- Incertidumbres estándar ---
    # Fuente 1: Temperatura del líquido (°F)
    u1 = np.sqrt(
        (0.072 / 2) ** 2 +
        (0.1 / np.sqrt(3)) ** 2 +
        (0.506 / np.sqrt(3)) ** 2 +
        (0.0457 / np.sqrt(3)) ** 2
    )  # ≈ 0.30185 °F

    # Fuente 2: Temperatura ambiente (°F)
    u2 = np.sqrt(
        (0.07 / 2) ** 2 +
        (0.1 / np.sqrt(3)) ** 2 +
        (0.506 / np.sqrt(3)) ** 2 +
        (1.5 / np.sqrt(3)) ** 2 +
        (0.0457 / np.sqrt(3)) ** 2
    )  # ≈ 0.95 °F

    # --- Coeficientes de sensibilidad ---
    s1 = 0.000010852  # para temperatura del líquido
    s2 = 0.000001550  # para temperatura ambiente

    # --- Simulación Monte Carlo ---
    fuente1_sim = np.random.normal(loc=Tl_nominal, scale=u1, size=n_sim)
    fuente2_sim = np.random.normal(loc=Tamb_nominal, scale=u2, size=n_sim)

    # Simulación de CTSh como suma ponderada de las fuentes por sensibilidad
    ctsh_sim = 1 + (s1 * (fuente1_sim - Tl_nominal)) + (s2 * (fuente2_sim - Tamb_nominal))

    # --- Estadísticas ---
    media = np.mean(ctsh_sim)
    uc = np.std(ctsh_sim, ddof=1)
    intervalo_95 = norm.interval(0.95, loc=media, scale=uc)
    k = (intervalo_95[1] - intervalo_95[0]) / (2 * uc)
    U = uc * k

    # Imprimir resultados con precisión
    print(f"🔷 Simulación Monte Carlo para CTSh")
    print(f"🔹 Incertidumbre típica combinada (uc): {uc:.10f}")
    print(f"🔹 Factor de cobertura efectivo (k): {k:.6f}")
    print(f"🔹 Incertidumbre expandida (U = uc · k): {U:.10f}")

    return ctsh_sim
