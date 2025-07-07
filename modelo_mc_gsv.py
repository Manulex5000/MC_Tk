# modelo_mc_gsv.py

import numpy as np
import math

Tref = 60.0
corr = 0.01374979547

K_tabla = {
    "crude oil": {
        "K0": 341.0957,
        "K1": 0.0,
        "K2": 0.0
    }
}

alpha_material = {
    "acero al carbon": 0.00000620,
    "inox 304": 0.00000961,
    "inox 316": 0.00000899,
    "monel": 0.00000720
}

def calcular_densidad_desde_API(api):
    g_agua = 999.016
    dens_rel = 141.5 / (api + 131.5)
    return dens_rel * g_agua

def calcular_CTSh(alfa, Tl, Tref):
    delta_T = Tl - Tref
    return 1 + 2 * alfa * delta_T + (alfa ** 2) * (delta_T ** 2)

def calcular_Bl(K0, K1, K2, dens):
    return K0 / (dens ** 2) + K1 / dens + K2

def calcular_CTL(Bl_value, Tl):
    delta_t = Tl - Tref
    exponent = -Bl_value * delta_t * (1 + 0.8 * Bl_value * (delta_t + corr))
    return math.exp(exponent)

def calcular_CSW(bsw):
    return 1 - bsw / 100

def modelo_montecarlo_gsv(n=100000, api=14.9, Tl=91.0, TOV=435.73, FW=0.0, material="acero al carbon", producto="crude oil", bsw=0.0):
    dens = calcular_densidad_desde_API(api)
    coef = K_tabla.get(producto.lower(), None)
    if coef is None:
        raise ValueError(f"Producto no encontrado en tabla: {producto}")
    K0, K1, K2 = coef["K0"], coef["K1"], coef["K2"]

    alfa = alpha_material.get(material.lower(), None)
    if alfa is None:
        raise ValueError(f"Material no reconocido: {material}")
    CTSh_nom = calcular_CTSh(alfa, Tl, Tref)
    Bl = calcular_Bl(K0, K1, K2, dens)
    CTL_nom = calcular_CTL(Bl, Tl)
    CSW_nom = calcular_CSW(bsw)

    u_TOV = 2.15 / 2.30
    u_FW = 2.16 / 3.0
    u_CTSh = 0.000007 / 2.0
    u_CTL = 0.00030 / 2.0

    TOV_sim = np.random.normal(TOV, u_TOV, n)
    FW_sim = np.random.normal(FW, u_FW, n)
    CTSh_sim = np.random.normal(CTSh_nom, u_CTSh, n)
    CTL_sim = np.random.normal(CTL_nom, u_CTL, n)
    CSW_sim = np.full(n, CSW_nom)  # constante, sin incertidumbre por ahora

    GSV_sim = (TOV_sim - FW_sim) * CTSh_sim * CTL_sim
    NSV_sim = GSV_sim * CSW_sim

    return NSV_sim, GSV_sim, CTSh_nom, CTL_nom, CSW_nom, dens
