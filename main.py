# main.py

from modelo_mc_gsv import modelo_montecarlo_gsv
from graficos import graficar_histograma
import numpy as np

# Entradas del modelo
simulaciones, GSV_sim, CTSh, CTL, CSW, dens = modelo_montecarlo_gsv(
    n=100000,
    api=14.9,
    Tl=91.0,
    TOV=435.73,
    FW=0.0,
    material="acero al carbon",
    producto="crude oil",
    bsw=0.0
)

# Cálculos estadísticos
media_NSV = np.mean(simulaciones)
desv_NSV = np.std(simulaciones, ddof=1)

# Resultados
print("✅ Resultados del modelo Monte Carlo")
print(f"CTSh = {CTSh:.8f}")
print(f"CTL  = {CTL:.6f}")
print(f"CSW  = {CSW:.6f}")
print(f"Densidad = {dens:.2f} kg/m³")
print(f"NSV Promedio = {media_NSV:.2f} Bbl")
print(f"U(NSV) (desviación estándar) = {desv_NSV:.2f} Bbl")

# Gráfico
graficar_histograma(simulaciones, variable="NSV", unidades="Bbl")
