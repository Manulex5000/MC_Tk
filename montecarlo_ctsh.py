import numpy as np
from scipy.stats import norm

def simulacion_ctsh(n_sim=100000):
    # Temperatura del líquido nominal
    Tl_nominal = 91.4  # °F
    Tamb_nominal = 87.0  # °F
    expansividad = 0.00001  # 1/°F (coef. típico)

    # ---- 1. Fuentes de incertidumbre - Temperatura del líquido ----
    u1 = np.sqrt(
        (0.072 / 2) ** 2 +
        (0.1 / np.sqrt(3)) ** 2 +
        (0.506 / np.sqrt(3)) ** 2 +
        (0.0457 / np.sqrt(3)) ** 2
    )  # resultado ≈ 0.30185 °F

    # ---- 2. Fuentes de incertidumbre - Temperatura ambiente ----
    u2 = np.sqrt(
        (0.07 / 2) ** 2 +
        (0.1 / np.sqrt(3)) ** 2 +
        (0.506 / np.sqrt(3)) ** 2 +
        (1.5 / np.sqrt(3)) ** 2 +
        (0.0457 / np.sqrt(3)) ** 2
    )  # resultado ≈ 0.95 °F

    # Simulación Monte Carlo
    Tl_sim = np.random.normal(loc=Tl_nominal, scale=u1, size=n_sim)
    Tamb_sim = np.random.normal(loc=Tamb_nominal, scale=u2, size=n_sim)

    deltaT = Tl_sim - Tamb_sim
    ctsh_sim = 1 + expansividad * deltaT

    # Estadísticos
    media = np.mean(ctsh_sim)
    uc = np.std(ctsh_sim, ddof=1)
    intervalo_95 = norm.interval(0.95, loc=media, scale=uc)
    k = (intervalo_95[1] - intervalo_95[0]) / (2 * uc)
    U = uc * k

    print("🔷 Simulación Monte Carlo para CTSh")
    print(f"🔹 Media del CTSh simulado: {round(media, 6)}")
    print(f"🔹 Incertidumbre típica combinada (uc): {round(uc, 6)}")
    print(f"🔹 Intervalo de confianza 95%: ({round(intervalo_95[0],6)} ; {round(intervalo_95[1],6)})")
    print(f"🔹 Factor de cobertura efectivo (k): {round(k, 6)}")
    print(f"🔹 Incertidumbre expandida (U = uc · k): {round(U, 6)}")

    return ctsh_sim
