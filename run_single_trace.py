"""
Single-trace PVIN run and printing.

Use this entry point when you want one simulation plus the state values at
a chosen time, without the tau/sigma sweep or plotting logic.
"""

import numpy as np
import matplotlib.pyplot as plt

from pvin_model import count_spikes, generate_ou_noise, run_pvin_with_ou

def main():
    mu = 0
    tau = 1000
    sigma = 1
    dt = 0.05
    T = 90000.0
    print_time = 3000.0

    t_noise, I_noise = generate_ou_noise(T, dt, mu, tau, sigma, seed=0)

    y0 = [-49.52776733833847,
          0.9827093136484906,
          0.024625396067553557,
          0.0022470251324288853,
          0.14110408589099907,
          0.03293826520477305,
          0.0971064287710309]
    Bt = 90.0

    sol = run_pvin_with_ou(t_noise, I_noise, Bt, y0)

    values_at_print_time = [np.interp(print_time, sol.t, sol.y[i]) for i in range(7)]
    V, h, n1, n3, Cai, r, m = values_at_print_time
    print(f"At t={print_time} ms (3.0 s):")
    print(f"V={V}, h={h}, n1={n1}, n3={n3}, Cai={Cai}, r={r}, m={m}")

    n_spikes = count_spikes(sol.t, sol.y[0])
    print(f"Total spikes detected: {n_spikes}")

    _, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(t_noise, I_noise, 'k', linewidth=0.5)
    axes[0].set_ylabel('I_noise (pA)')
    axes[0].set_title('OU noise input current')

    axes[1].plot(sol.t, sol.y[0], 'k', linewidth=0.5)
    axes[1].set_xlabel('T (ms)')
    axes[1].set_ylabel('V (mV)')
    axes[1].set_title('PVIN response to OU noise')
    axes[1].set_ylim(-80, 50)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()