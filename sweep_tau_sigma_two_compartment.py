"""
Tau/sigma sweep analysis and plotting for the two-compartment (soma + AIS)
PVIN model.

Each (tau, sigma) combination is run over multiple independent OU noise
seeds (n_trials), and the mean spike count is reported/plotted.
"""

import numpy as np
import matplotlib.pyplot as plt

from pvin_model import (
    count_spikes,
    default_two_compartment_initial_state,
    generate_ou_noise,
    run_pvin_two_compartment_with_ou,
)


def run_tau_sigma_sweep_two_compartment(tau_values, sigma_values, g_c, kappa,
                                         T=90000.0, dt=0.05, mu=0.0, Bt=90.0,
                                         n_trials=5, seed0=0, y0=None,
                                         spike_threshold=-20.0, min_isi=2.0):
    """
    Sweep OU noise tau and sigma values for the two-compartment (soma+AIS)
    model, averaging spike counts over n_trials independent noise
    realizations at each combination.

    Parameters
    ----------
    tau_values : sequence of float
        OU correlation time constants to test (ms).
    sigma_values : sequence of float
        OU noise amplitudes to test (pA).
    g_c : float
        Axial coupling conductance between soma and AIS (nS).
    kappa : float
        Soma-to-total surface area ratio (0 < kappa < 1).
    T : float, optional
        Simulation duration (ms).
    dt : float, optional
        OU noise integration timestep (ms).
    mu : float, optional
        OU mean current (pA).
    Bt : float, optional
        Calcium buffer capacity (uM).
    n_trials : int, optional
        Number of independent noise realizations (seeds) averaged at each
        (tau, sigma) combination.
    seed0 : int, optional
        Base random seed; trial i uses seed = seed0 + i.
    y0 : array_like, shape (16,)
        Initial condition [soma states..., AIS states...]. Required.
    spike_threshold : float, optional
        Voltage threshold used for spike counting (mV).
    min_isi : float, optional
        Minimum inter-spike interval used for spike counting (ms).

    Returns
    -------
    soma_mean : ndarray, shape (len(tau_values), len(sigma_values))
        Mean soma spike count across trials, for each (tau, sigma).
    soma_std : ndarray, same shape
        Std. dev. of soma spike count across trials.
    ais_mean : ndarray, same shape
        Mean AIS spike count across trials.
    ais_std : ndarray, same shape
        Std. dev. of AIS spike count across trials.
    """

    if y0 is None:
        y0 = default_two_compartment_initial_state()

    soma_mean = np.zeros((len(tau_values), len(sigma_values)))
    soma_std = np.zeros((len(tau_values), len(sigma_values)))
    ais_mean = np.zeros((len(tau_values), len(sigma_values)))
    ais_std = np.zeros((len(tau_values), len(sigma_values)))

    for i, tau in enumerate(tau_values):
        for j, sigma in enumerate(sigma_values):
            soma_trials = np.zeros(n_trials)
            ais_trials = np.zeros(n_trials)
            for k in range(n_trials):
                t_noise, I_OU = generate_ou_noise(T, dt, mu, tau, sigma, seed=seed0 + k)
                sol = run_pvin_two_compartment_with_ou(t_noise, I_OU, Bt, y0, g_c, kappa)
                soma_trials[k] = count_spikes(
                    sol.t, sol.y[0], threshold=spike_threshold, min_isi=min_isi
                )
                ais_trials[k] = count_spikes(
                    sol.t, sol.y[8], threshold=spike_threshold, min_isi=min_isi
                )

            soma_mean[i, j] = soma_trials.mean()
            soma_std[i, j] = soma_trials.std()
            ais_mean[i, j] = ais_trials.mean()
            ais_std[i, j] = ais_trials.std()
            print(f"tau={tau:>6.1f} ms, sigma={sigma:.2f} pA -> "
                  f"soma: {soma_mean[i, j]:.1f}+/-{soma_std[i, j]:.1f} spikes, "
                  f"AIS: {ais_mean[i, j]:.1f}+/-{ais_std[i, j]:.1f} spikes "
                  f"(n={n_trials})")

    return soma_mean, soma_std, ais_mean, ais_std


def plot_tau_sigma_sweep(tau_values, sigma_values, spike_counts_mean, spike_counts_std=None,
                          compartment_label="soma"):
    """Plot mean spike count curves by tau and a tau/sigma heatmap.

    Parameters
    ----------
    spike_counts_std : ndarray, optional
        Std. dev. across trials, same shape as spike_counts_mean. If
        given, shown as error bars on the line plot.
    compartment_label : str, optional
        Used in plot titles/print statements, e.g. "soma" or "AIS".
    """
    sigma_arr = np.asarray(sigma_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, tau in enumerate(tau_values):
        yerr = spike_counts_std[i] if spike_counts_std is not None else None
        axes[0].errorbar(sigma_arr, spike_counts_mean[i], yerr=yerr,
                          marker='o', capsize=3, label=f'tau={tau} ms')

        if len(sigma_arr) > 1 and np.std(spike_counts_mean[i]) > 0:
            corr = np.corrcoef(sigma_arr, spike_counts_mean[i])[0, 1]
        else:
            corr = float('nan')
        print(f"[{compartment_label}] tau={tau} ms: correlation(sigma, mean spike count) = {corr:.3f}")

    axes[0].set_xlabel('sigma (pA)')
    axes[0].set_ylabel('mean spike count')
    axes[0].set_title(f'Mean spike count vs sigma, by tau ({compartment_label})')
    axes[0].legend()

    im = axes[1].imshow(
        spike_counts_mean,
        aspect='auto',
        origin='lower',
        extent=[sigma_arr[0], sigma_arr[-1], 0, len(tau_values)],
    )
    axes[1].set_yticks(np.arange(len(tau_values)) + 0.5)
    axes[1].set_yticklabels([f'{tau}' for tau in tau_values])
    axes[1].set_xlabel('sigma (pA)')
    axes[1].set_ylabel('tau (ms)')
    axes[1].set_title(f'Mean spike count heatmap ({compartment_label})')
    fig.colorbar(im, ax=axes[1], label='mean spike count')

    plt.tight_layout()
    plt.show()


def main():
    mu = 0
    dt = 0.05
    T = 90000.0
    Bt = 90.0

    g_c = 0.09
    kappa = 0.9
    n_trials = 5

    y0 = default_two_compartment_initial_state()

    tau_values = [10, 100, 1000]
    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    soma_mean, soma_std, ais_mean, ais_std = run_tau_sigma_sweep_two_compartment(
        tau_values, sigma_values, g_c, kappa, T=T, dt=dt, mu=mu, Bt=Bt,
        n_trials=n_trials, y0=y0,
    )
    plot_tau_sigma_sweep(tau_values, sigma_values, soma_mean, soma_std,
                          compartment_label="soma")
    plot_tau_sigma_sweep(tau_values, sigma_values, ais_mean, ais_std,
                          compartment_label="AIS")


if __name__ == "__main__":
    main()