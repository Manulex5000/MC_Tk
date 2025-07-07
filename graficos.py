# MC2/graficos.py

import matplotlib.pyplot as plt
import numpy as np

def graficar_histograma(simulaciones, variable="GSV", unidades="Bbl", bins=50):
    """
    Grafica un histograma de las simulaciones Monte Carlo.
    """
    media = np.mean(simulaciones)
    std = np.std(simulaciones, ddof=1)

    plt.figure(figsize=(10, 6))
    plt.hist(simulaciones, bins=bins, color="#4A90E2", edgecolor='black', alpha=0.7)
    plt.axvline(media, color='red', linestyle='--', label=f"Media: {media:.2f} {unidades}")
    plt.axvline(media - 2*std, color='green', linestyle=':', label=f"-2σ: {media - 2*std:.2f}")
    plt.axvline(media + 2*std, color='green', linestyle=':', label=f"+2σ: {media + 2*std:.2f}")

    plt.title(f"Histograma de Simulaciones Monte Carlo - {variable}", fontsize=16)
    plt.xlabel(f"{variable} ({unidades})", fontsize=14)
    plt.ylabel("Frecuencia", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()