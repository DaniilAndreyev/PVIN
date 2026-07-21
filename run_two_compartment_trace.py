"""
Two-compartment (soma + AIS) PVIN trace run, printing, and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt

from pvin_model import (
    count_spikes,
    default_two_compartment_initial_state,
    generate_ou_noise,
    run_pvin_two_compartment_with_ou,
)


def main():
    mu = 0
    tau = 1000
    sigma = 1
    dt = 0.05
    T = 90000.0
    print_time = 3000.0

    g_c = 0.09
    kappa = 0.9

    t_noise, I_noise = generate_ou_noise(T, dt, mu, tau, sigma, seed=0)

    y0 = default_two_compartment_initial_state()
    Bt = 90.0

    sol = run_pvin_two_compartment_with_ou(t_noise, I_noise, Bt, y0, g_c, kappa)

    values_at_print_time = [np.interp(print_time, sol.t, sol.y[i]) for i in range(14)]
    V, h, n1, n3, Cai, r, m, V_AIS, h_AIS, n1_AIS, n3_AIS, Cai_AIS, r_AIS, m_AIS = values_at_print_time
    print(f"At t={print_time} ms (3.0 s):")
    print(f"Soma: V={V}, h={h}, n1={n1}, n3={n3}, Cai={Cai}, r={r}, m={m}")
    print(f"AIS:  V={V_AIS}, h={h_AIS}, n1={n1_AIS}, n3={n3_AIS}, Cai={Cai_AIS}, r={r_AIS}, m={m_AIS}")

    n_spikes_soma = count_spikes(sol.t, sol.y[0])
    n_spikes_ais = count_spikes(sol.t, sol.y[7])
    print(f"Total soma spikes detected: {n_spikes_soma}")
    print(f"Total AIS spikes detected: {n_spikes_ais}")

    _, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(t_noise, I_noise, 'k', linewidth=0.5)
    axes[0].set_ylabel('I_noise (pA)')
    axes[0].set_title('OU noise input current')

    axes[1].plot(sol.t, sol.y[0], linewidth=0.7, label='Soma')
    axes[1].plot(sol.t, sol.y[7], linewidth=0.7, label='AIS')
    axes[1].set_xlabel('T (ms)')
    axes[1].set_ylabel('V (mV)')
    axes[1].set_title('Soma vs AIS response to OU noise')
    axes[1].set_ylim(-80, 50)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()