# montecarlo_ctl.py
import numpy as np
import matplotlib.pyplot as plt

def simulacion_incertidumbre_ctl(n_sim=1000000):
    # Valores nominales
    Tl_nominal = 91.4       # °F
    API_60 = 14.9

    # Incertidumbres estándar
    u_temp = np.sqrt(
        (0.072 / 2)**2 + 
        (0.1 / np.sqrt(3))**2 + 
        (0.506 / np.sqrt(3))**2 + 
        ((0.0005 * Tl_nominal) / np.sqrt(3))**2
    )  # ≈ 0.30185 °F

    u_api = (((API_60 + 131.5)**2) / 141500 * 5) / np.sqrt(3)  # ≈ 0.437

    # Coeficientes de sensibilidad
    sens_temp = -0.00040
    sens_api = -0.00020

    # Simulaciones Monte Carlo
    temp_sim = np.random.normal(loc=Tl_nominal, scale=u_temp, size=n_sim)
    api_sim = np.random.normal(loc=API_60, scale=u_api, size=n_sim)

    # Perturbación total del CTL a partir de las dos fuentes
    perturbacion_ctl = (temp_sim - Tl_nominal) * sens_temp + (api_sim - API_60) * sens_api

    # Media debe estar centrada en 0 porque las simulaciones son relativas
    uc_ctl = np.std(perturbacion_ctl, ddof=1)  # Incertidumbre combinada
    k = 2  # Factor de cobertura para 95%
    u_expandida = uc_ctl * k

    # Mostrar resultados
    print("🔷 Simulación Monte Carlo para CTL")
    print(f"🔹 Incertidumbre típica combinada del CTL: {round(uc_ctl, 6)}")
    print(f"🔹 Incertidumbre expandida del CTL (k=2): {round(u_expandida, 6)}")

    return perturbacion_ctl, u_expandida, uc_ctl, k

def graficar_ctl(perturbacion_ctl):
    plt.hist(perturbacion_ctl, bins=100, density=True, color="salmon", edgecolor="black")
    plt.title("Distribución simulada del CTL (Monte Carlo)")
    plt.xlabel("Perturbación del CTL")
    plt.ylabel("Densidad")
    plt.grid(True)
    plt.axvline(np.mean(perturbacion_ctl), color="red", linestyle="--", label=f"Media: {np.mean(perturbacion_ctl):.6f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Ejecutar directamente si es llamado como script
if __name__ == "__main__":
    perturbacion_ctl, u_expandida, k = simulacion_incertidumbre_ctl()
    graficar_ctl(perturbacion_ctl)