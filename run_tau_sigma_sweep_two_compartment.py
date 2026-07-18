"""
Tau/sigma sweep analysis and plotting for the PVIN model.

Includes both the single-compartment sweep (run_tau_sigma_sweep) and the
two-compartment soma+AIS sweep (run_tau_sigma_sweep_two_compartment).
"""

import numpy as np
import matplotlib.pyplot as plt

from pvin_model import (
    count_spikes,
    default_soma_initial_state,
    default_two_compartment_initial_state,
    generate_ou_noise,
    run_pvin_with_ou,
    run_pvin_two_compartment_with_ou,
)


def run_tau_sigma_sweep(tau_values, sigma_values, T=90000.0, dt=0.05,
                         mu=0.0, Bt=90.0, seed=0,
                         y0=None, spike_threshold=-20.0, min_isi=2.0):
    spike_counts = np.zeros((len(tau_values), len(sigma_values)), dtype=int)

    for i, tau in enumerate(tau_values):
        for j, sigma in enumerate(sigma_values):
            t_noise, I_OU = generate_ou_noise(T, dt, mu, tau, sigma, seed=seed)
            sol = run_pvin_with_ou(t_noise, I_OU, Bt, y0)
            n_spikes = count_spikes(
                sol.t,
                sol.y[0],
                threshold=spike_threshold,
                min_isi=min_isi,
            )
            spike_counts[i, j] = n_spikes
            print(f"tau={tau:>6.1f} ms, sigma={sigma:.2f} pA -> {n_spikes} spikes")

    return spike_counts


def run_tau_sigma_sweep_two_compartment(tau_values, sigma_values, g_c, kappa,
                                         T=90000.0, dt=0.05, mu=0.0, Bt=90.0,
                                         seed=0, y0=None,
                                         spike_threshold=-20.0, min_isi=2.0):

    soma_spike_counts = np.zeros((len(tau_values), len(sigma_values)), dtype=int)
    ais_spike_counts = np.zeros((len(tau_values), len(sigma_values)), dtype=int)

    for i, tau in enumerate(tau_values):
        for j, sigma in enumerate(sigma_values):
            t_noise, I_OU = generate_ou_noise(T, dt, mu, tau, sigma, seed=seed)
            sol = run_pvin_two_compartment_with_ou(t_noise, I_OU, Bt, y0, g_c, kappa)
            n_spikes_soma = count_spikes(
                sol.t, sol.y[0], threshold=spike_threshold, min_isi=min_isi
            )
            n_spikes_ais = count_spikes(
                sol.t, sol.y[7], threshold=spike_threshold, min_isi=min_isi
            )
            soma_spike_counts[i, j] = n_spikes_soma
            ais_spike_counts[i, j] = n_spikes_ais
            print(f"tau={tau:>6.1f} ms, sigma={sigma:.2f} pA -> "
                  f"soma: {n_spikes_soma} spikes, AIS: {n_spikes_ais} spikes")

    return soma_spike_counts, ais_spike_counts


def plot_tau_sigma_sweep(tau_values, sigma_values, spike_counts,
                          compartment_label="soma"):
    sigma_arr = np.asarray(sigma_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, tau in enumerate(tau_values):
        axes[0].plot(sigma_arr, spike_counts[i], marker='o', label=f'tau={tau} ms')

        if len(sigma_arr) > 1 and np.std(spike_counts[i]) > 0:
            corr = np.corrcoef(sigma_arr, spike_counts[i])[0, 1]
        else:
            corr = float('nan')
        print(f"[{compartment_label}] tau={tau} ms: correlation(sigma, spike count) = {corr:.3f}")

    axes[0].set_xlabel('sigma (pA)')
    axes[0].set_ylabel('spike count')
    axes[0].set_title(f'Spike count vs sigma, by tau ({compartment_label})')
    axes[0].legend()

    im = axes[1].imshow(
        spike_counts,
        aspect='auto',
        origin='lower',
        extent=[sigma_arr[0], sigma_arr[-1], 0, len(tau_values)],
    )
    axes[1].set_yticks(np.arange(len(tau_values)) + 0.5)
    axes[1].set_yticklabels([f'{tau}' for tau in tau_values])
    axes[1].set_xlabel('sigma (pA)')
    axes[1].set_ylabel('tau (ms)')
    axes[1].set_title(f'Spike count heatmap ({compartment_label})')
    fig.colorbar(im, ax=axes[1], label='spike count')

    plt.tight_layout()
    plt.show()


def main():
    mu = 0
    dt = 0.05
    T = 90000.0
    Bt = 90.0

    y0_soma = default_soma_initial_state()

    tau_values = [10, 100, 1000]
    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # Single-compartment sweep
    spike_counts = run_tau_sigma_sweep(tau_values, sigma_values, T=T, dt=dt,
                                        mu=mu, Bt=Bt, y0=y0_soma)
    plot_tau_sigma_sweep(tau_values, sigma_values, spike_counts)

    # Two-compartment sweep
    g_c = 5.0
    kappa = 0.3
    y0_two_compartment = default_two_compartment_initial_state()

    soma_spike_counts, ais_spike_counts = run_tau_sigma_sweep_two_compartment(
        tau_values, sigma_values, g_c, kappa, T=T, dt=dt, mu=mu, Bt=Bt,
        y0=y0_two_compartment,
    )
    plot_tau_sigma_sweep(tau_values, sigma_values, soma_spike_counts,
                          compartment_label="soma")
    plot_tau_sigma_sweep(tau_values, sigma_values, ais_spike_counts,
                          compartment_label="AIS")


if __name__ == "__main__":
    main()