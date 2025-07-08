import numpy as np
import math
import matplotlib.pyplot as plt

# Importar incertidumbres y simulaciones desde otros módulos
from montecarlo_tov import simulacion_tov

# Ejecutar simulación Monte Carlo del TOV
tov_simulado, u_exp_tov, k_tov = simulacion_tov()

# Ejecutar simulación Monte Carlo del FW
from montecarlo_fw import simulacion_fw
fw_simulado, u_exp_fw, k_fw = simulacion_fw()

# Ejecutar simulación Monte Carlo del CTL
from montecarlo_ctl import simulacion_incertidumbre_ctl
perturbacion_ctl, u_exp_ctl, uc_ctl, k_ctl = simulacion_incertidumbre_ctl()

# Ejecutar simulación Monte Carlo del CTSh
from montecarlo_ctsh import simulacion_ctsh
ctsh_sim, u_exp_ctsh, uc_ctsh, k_ctsh = simulacion_ctsh()


Tref = 60.0
corr = 0.01374979547

# Tabla de coeficientes para cálculo de Bl
K_tabla = {
    "crude oil": {
        "K0": 341.0957,
        "K1": 0.0,
        "K2": 0.0
    }
}

# Coeficientes de expansión térmica por material
alpha_material = {
    "acero al carbon": 0.00000620,
    "inox 304": 0.00000961,
    "inox 316": 0.00000899,
    "monel": 0.00000720
}

# Función para obtener densidad del petróleo desde API
def calcular_densidad_desde_API(api):
    g_agua = 999.016  # kg/m³ a 60°F
    dens_rel = 141.5 / (api + 131.5)
    return dens_rel * g_agua  # kg/m³

# Factor de corrección por expansión térmica del tanque
def calcular_CTSh(alfa, Tl, Tref):
    delta_T = Tl - Tref
    return 1 + 2 * alfa * delta_T + (alfa ** 2) * (delta_T ** 2)

# Cálculo del coeficiente Bl
def calcular_Bl(K0, K1, K2, dens):
    return K0 / (dens ** 2) + K1 / dens + K2

# Cálculo del CTL desde Bl y temperatura líquida
def calcular_CTL(Bl_value, Tl):
    delta_t = Tl - Tref
    exponent = -Bl_value * delta_t * (1 + 0.8 * Bl_value * (delta_t + corr))
    return math.exp(exponent)

# Cálculo del Correction for Sediment and Water
def calcular_CSW(bsw):
    return 1 - bsw / 100

# Modelo Monte Carlo para estimar GSV y NSV
def modelo_montecarlo_gsv(
    n=1000000,
    api=14.9,
    Tl=91.0,
    TOV=435.73,
    FW=0.0,
    material="acero al carbon",
    producto="crude oil",
    bsw=0.0
):
    # Calcular incertidumbres estándar desde cada módulo
    u_TOV = u_exp_tov / k_tov
    u_FW = u_exp_fw / k_fw
    u_CTSh = u_exp_ctsh / k_ctsh
    u_CTL = u_exp_ctl / k_ctl

    # Cálculo de densidad del crudo
    dens = calcular_densidad_desde_API(api)

    # Obtener coeficientes Bl del producto
    coef = K_tabla.get(producto.lower())
    if coef is None:
        raise ValueError(f"Producto no encontrado en tabla: {producto}")
    K0, K1, K2 = coef["K0"], coef["K1"], coef["K2"]

    # Obtener coeficiente de expansión térmica
    alfa = alpha_material.get(material.lower())
    if alfa is None:
        raise ValueError(f"Material no reconocido: {material}")

    # Valores nominales
    CTSh_nom = calcular_CTSh(alfa, Tl, Tref)
    Bl = calcular_Bl(K0, K1, K2, dens)
    CTL_nom = calcular_CTL(Bl, Tl)
    CSW_nom = calcular_CSW(bsw)

    # Simulaciones Monte Carlo para cada variable
    TOV_sim = np.random.normal(TOV, u_TOV, n)
    FW_sim = np.random.normal(FW, u_FW, n)
    CTSh_sim = np.random.normal(CTSh_nom, u_CTSh, n)
    CTL_sim = np.random.normal(CTL_nom, u_CTL, n)

    # Cálculo de GSV y NSV simulados
    GSV_sim = (TOV_sim - FW_sim) * CTSh_sim * CTL_sim
    NSV_sim = GSV_sim * CSW_nom

    # Estadísticas globales del GSV
    media_GSV = np.mean(GSV_sim)
    u_std_global = np.std(GSV_sim, ddof=1)
    p2_5, p97_5 = np.percentile(GSV_sim, [2.5, 97.5])
    u_exp_global = (p97_5 - p2_5) / 2
    k_global = u_exp_global / u_std_global

    return {
        "NSV_simulado": NSV_sim,
        "GSV_simulado": GSV_sim,
        "CTSh_nominal": CTSh_nom,
        "CTL_nominal": CTL_nom,
        "CSW_nominal": CSW_nom,
        "densidad_API_60F": dens,
        "media_GSV": media_GSV,
        "u_std_global": u_std_global,
        "u_exp_global": u_exp_global,
        "k_global": k_global
    }

def graficar_nsv(NSV_sim):
    plt.figure(figsize=(10, 5))
    plt.hist(NSV_sim, bins=100, density=True, color='lightgreen', edgecolor='black')
    plt.title("Distribución simulada del NSV (Monte Carlo)")
    plt.xlabel("NSV [Bbl]")
    plt.ylabel("Densidad")
    plt.grid(True)
    plt.axvline(np.mean(NSV_sim), color='red', linestyle='--', label=f"Media: {np.mean(NSV_sim):.2f} Bbl")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    resultados = modelo_montecarlo_gsv()
    graficar_nsv(resultados["NSV_simulado"])
