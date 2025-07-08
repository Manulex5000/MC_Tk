import numpy as np
from montecarlo_tov import tov_simulado, k_mc as k_tov
from montecarlo_fw import fw_simulado, k as k_fw
from montecarlo_ctl import ctl_simulado  # k = 2
from montecarlo_ctsh import ctsh_simulado, k as k_ctsh

# Paso 1: Validar que todos los arrays tengan la misma longitud
N = len(tov_simulado)
assert all(len(arr) == N for arr in [fw_simulado, ctl_simulado, ctsh_simulado]), \
    "Los vectores simulados no tienen la misma cantidad de iteraciones"

# Paso 2: Convertir a arrays de NumPy
tov_simulado = np.array(tov_simulado)
fw_simulado = np.array(fw_simulado)
ctl_simulado = np.array(ctl_simulado)
ctsh_simulado = np.array(ctsh_simulado)

# Paso 3: Calcular GSV simulado
gsv_simulado = (tov_simulado - fw_simulado) * ctsh_simulado * ctl_simulado

# Paso 4: Calcular estadísticos de la distribución de GSV
media_gsv = np.mean(gsv_simulado)
u_std_gsv = np.std(gsv_simulado, ddof=1)

# Paso 5: Calcular percentiles 2.5% y 97.5% para el intervalo del 95%
p2_5 = np.percentile(gsv_simulado, 2.5)
p97_5 = np.percentile(gsv_simulado, 97.5)
U_exp_gsv = (p97_5 - p2_5) / 2  # Incertidumbre expandida MC

# Paso 6: Calcular el factor de cobertura efectivo
k_mc_global = U_exp_gsv / u_std_gsv

# Paso 7: Mostrar resultados
print("========== Resultados Globales del GSV por Monte Carlo ==========")
print(f"Valor medio simulado GSV:            {media_gsv:.6f}")
print(f"Incertidumbre estándar (u_std):      {u_std_gsv:.6f}")
print(f"Incertidumbre expandida (95%):       {U_exp_gsv:.6f}")
print(f"Factor de cobertura efectivo (k_mc): {k_mc_global:.3f}")
print(f"Intervalo [2.5%, 97.5%]:             [{p2_5:.6f}, {p97_5:.6f}]")
print("==================================================================")

# (Opcional) Estructura para exportar como JSON si lo necesitas
resultados_gsv = {
    "media_gsv": media_gsv,
    "u_std_gsv": u_std_gsv,
    "U_exp_gsv": U_exp_gsv,
    "k_mc_global": k_mc_global,
    "percentil_2_5": p2_5,
    "percentil_97_5": p97_5
}
