"""
Sigma vs Btot (calcium buffer capacity) sweep for the two-compartment
(soma + AIS) PVIN model, holding tau fixed.
"""

import numpy as np
import matplotlib.pyplot as plt

from pvin_model import (
    count_spikes,
    default_two_compartment_initial_state,
    generate_ou_noise,
    run_pvin_two_compartment_with_ou,
)


def run_sigma_Bt_sweep_two_compartment(sigma_values, Bt_values, tau, g_c, kappa,
                                        T=90000.0, dt=0.05, mu=0.0,
                                        seed=0, y0=None,
                                        spike_threshold=-20.0, min_isi=2.0):
    """
    Sweep OU noise sigma and calcium buffer capacity (Bt) for the
    two-compartment model, at a fixed tau, counting spikes separately
    for each compartment at every combination.

    Parameters
    ----------
    sigma_values : sequence of float
        OU noise amplitudes to test (pA).
    Bt_values : sequence of float
        Calcium buffer capacities to test (uM).
    tau : float
        OU correlation time constant, held fixed across the sweep (ms).
    g_c : float
        Axial coupling conductance between soma and AIS (nS).
    kappa : float
        Soma-to-total surface area ratio (0 < kappa < 1).
    T, dt, mu, seed, spike_threshold, min_isi : see run_tau_sigma_sweep.
    y0 : array_like, shape (14,)
        Initial condition [soma states..., AIS states...]. Required.

    Returns
    -------
    soma_spike_counts : ndarray, shape (len(sigma_values), len(Bt_values))
        Soma spike counts for each (sigma, Bt) combination.
    ais_spike_counts : ndarray, shape (len(sigma_values), len(Bt_values))
        AIS spike counts for each (sigma, Bt) combination.
    """

    if y0 is None:
        y0 = default_two_compartment_initial_state()

    soma_spike_counts = np.zeros((len(sigma_values), len(Bt_values)), dtype=int)
    ais_spike_counts = np.zeros((len(sigma_values), len(Bt_values)), dtype=int)

    for i, sigma in enumerate(sigma_values):
        t_noise, I_OU = generate_ou_noise(T, dt, mu, tau, sigma, seed=seed)
        for j, Bt in enumerate(Bt_values):
            sol = run_pvin_two_compartment_with_ou(t_noise, I_OU, Bt, y0, g_c, kappa)
            n_spikes_soma = count_spikes(
                sol.t, sol.y[0], threshold=spike_threshold, min_isi=min_isi
            )
            n_spikes_ais = count_spikes(
                sol.t, sol.y[7], threshold=spike_threshold, min_isi=min_isi
            )
            soma_spike_counts[i, j] = n_spikes_soma
            ais_spike_counts[i, j] = n_spikes_ais
            print(f"sigma={sigma:.2f} pA, Bt={Bt:.1f} uM -> "
                  f"soma: {n_spikes_soma} spikes, AIS: {n_spikes_ais} spikes")

    return soma_spike_counts, ais_spike_counts


def plot_sigma_Bt_sweep(sigma_values, Bt_values, spike_counts, compartment_label="soma"):
    """Plot spike count curves by sigma (vs Bt) and a sigma/Bt heatmap."""
    Bt_arr = np.asarray(Bt_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, sigma in enumerate(sigma_values):
        axes[0].plot(Bt_arr, spike_counts[i], marker='o', label=f'sigma={sigma} pA')

        if len(Bt_arr) > 1 and np.std(spike_counts[i]) > 0:
            corr = np.corrcoef(Bt_arr, spike_counts[i])[0, 1]
        else:
            corr = float('nan')
        print(f"[{compartment_label}] sigma={sigma} pA: correlation(Bt, spike count) = {corr:.3f}")

    axes[0].set_xlabel('Bt (uM)')
    axes[0].set_ylabel('spike count')
    axes[0].set_title(f'Spike count vs Bt, by sigma ({compartment_label})')
    axes[0].legend()

    im = axes[1].imshow(
        spike_counts,
        aspect='auto',
        origin='lower',
        extent=[Bt_arr[0], Bt_arr[-1], 0, len(sigma_values)],
    )
    axes[1].set_yticks(np.arange(len(sigma_values)) + 0.5)
    axes[1].set_yticklabels([f'{sigma}' for sigma in sigma_values])
    axes[1].set_xlabel('Bt (uM)')
    axes[1].set_ylabel('sigma (pA)')
    axes[1].set_title(f'Spike count heatmap ({compartment_label})')
    fig.colorbar(im, ax=axes[1], label='spike count')

    plt.tight_layout()
    plt.show()


def main():
    mu = 0
    dt = 0.05
    T = 90000.0

    g_c = 0.09
    kappa = 0.9
    tau = 100

    y0 = default_two_compartment_initial_state()

    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    Bt_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    soma_spike_counts, ais_spike_counts = run_sigma_Bt_sweep_two_compartment(
        sigma_values, Bt_values, tau, g_c, kappa, T=T, dt=dt, mu=mu, y0=y0,
    )
    plot_sigma_Bt_sweep(sigma_values, Bt_values, soma_spike_counts, compartment_label="soma")
    plot_sigma_Bt_sweep(sigma_values, Bt_values, ais_spike_counts, compartment_label="AIS")


if __name__ == "__main__":
    main()