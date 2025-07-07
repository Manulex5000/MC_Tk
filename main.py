from montecarlo_tov import simulacion_tov
from montecarlo_fw import simulacion_fw
from montecarlo_ctl import simulacion_incertidumbre_ctl
from montecarlo_ctsh import simulacion_ctsh

def main():
    print("🔷 Simulación Monte Carlo para TOV")
    tov_simulado = simulacion_tov(n_sim=1000000)

    print("\n" + "-"*60 + "\n")

    print("🔷 Simulación Monte Carlo para FW")
    fw_simulado = simulacion_fw(n_sim=1000000)

    print("\n" + "-"*60 + "\n")

    print("🔷 Simulación Monte Carlo para CTL")
    ctl_simulado = simulacion_incertidumbre_ctl(n_sim=1000000)

    print("\n" + "-"*60 + "\n")

    print("🔷 Simulación Monte Carlo para CTSh")
    ctsh_simulado = simulacion_ctsh(n_sim=1000000)

if __name__ == "__main__":
    main()
