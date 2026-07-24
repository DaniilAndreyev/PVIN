"""
Tau vs Btot (calcium buffer capacity) sweep for the two-compartment
(soma + AIS) PVIN model, holding sigma fixed.

Each (tau, Bt) combination is run over multiple independent OU noise
seeds (n_trials), and the mean spike count is reported/plotted, per
Fritz's note that a single noise realization is not representative.
"""

import numpy as np
import matplotlib.pyplot as plt

from pvin_model import (
    count_spikes,
    default_two_compartment_initial_state,
    generate_ou_noise,
    run_pvin_two_compartment_with_ou,
)


def run_tau_Bt_sweep_two_compartment(tau_values, Bt_values, sigma, g_c, kappa,
                                      T=90000.0, dt=0.05, mu=0.0,
                                      n_trials=5, seed0=0, y0=None,
                                      spike_threshold=-20.0, min_isi=2.0):
    """
    Sweep OU noise tau and calcium buffer capacity (Bt) for the
    two-compartment model, at a fixed sigma, averaging spike counts over
    n_trials independent noise realizations at each combination.

    Parameters
    ----------
    tau_values : sequence of float
        OU correlation time constants to test (ms).
    Bt_values : sequence of float
        Calcium buffer capacities to test (uM).
    sigma : float
        OU noise amplitude, held fixed across the sweep (pA).
    g_c : float
        Axial coupling conductance between soma and AIS (nS).
    kappa : float
        Soma-to-total surface area ratio (0 < kappa < 1).
    T, dt, mu : see run_tau_sigma_sweep_two_compartment.
    n_trials : int, optional
        Number of independent noise realizations (seeds) averaged at each
        (tau, Bt) combination.
    seed0 : int, optional
        Base random seed; trial i uses seed = seed0 + i.
    y0 : array_like, shape (16,)
        Initial condition [soma states..., AIS states...]. Required.
    spike_threshold, min_isi : see run_tau_sigma_sweep_two_compartment.

    Returns
    -------
    soma_mean : ndarray, shape (len(tau_values), len(Bt_values))
        Mean soma spike count across trials, for each (tau, Bt).
    soma_std : ndarray, same shape
        Std. dev. of soma spike count across trials.
    ais_mean : ndarray, same shape
        Mean AIS spike count across trials.
    ais_std : ndarray, same shape
        Std. dev. of AIS spike count across trials.
    """

    if y0 is None:
        y0 = default_two_compartment_initial_state()

    soma_mean = np.zeros((len(tau_values), len(Bt_values)))
    soma_std = np.zeros((len(tau_values), len(Bt_values)))
    ais_mean = np.zeros((len(tau_values), len(Bt_values)))
    ais_std = np.zeros((len(tau_values), len(Bt_values)))

    for i, tau in enumerate(tau_values):
        for j, Bt in enumerate(Bt_values):
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
            print(f"tau={tau:>6.1f} ms, Bt={Bt:.1f} uM -> "
                  f"soma: {soma_mean[i, j]:.1f}+/-{soma_std[i, j]:.1f} spikes, "
                  f"AIS: {ais_mean[i, j]:.1f}+/-{ais_std[i, j]:.1f} spikes "
                  f"(n={n_trials})")

    return soma_mean, soma_std, ais_mean, ais_std


def plot_tau_Bt_sweep(tau_values, Bt_values, spike_counts_mean, spike_counts_std=None,
                       compartment_label="soma"):
    """Plot mean spike count curves by tau (vs Bt) and a tau/Bt heatmap."""
    Bt_arr = np.asarray(Bt_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, tau in enumerate(tau_values):
        yerr = spike_counts_std[i] if spike_counts_std is not None else None
        axes[0].errorbar(Bt_arr, spike_counts_mean[i], yerr=yerr,
                          marker='o', capsize=3, label=f'tau={tau} ms')

        if len(Bt_arr) > 1 and np.std(spike_counts_mean[i]) > 0:
            corr = np.corrcoef(Bt_arr, spike_counts_mean[i])[0, 1]
        else:
            corr = float('nan')
        print(f"[{compartment_label}] tau={tau} ms: correlation(Bt, mean spike count) = {corr:.3f}")

    axes[0].set_xlabel('Bt (uM)')
    axes[0].set_ylabel('mean spike count')
    axes[0].set_title(f'Mean spike count vs Bt, by tau ({compartment_label})')
    axes[0].legend()

    im = axes[1].imshow(
        spike_counts_mean,
        aspect='auto',
        origin='lower',
        extent=[Bt_arr[0], Bt_arr[-1], 0, len(tau_values)],
    )
    axes[1].set_yticks(np.arange(len(tau_values)) + 0.5)
    axes[1].set_yticklabels([f'{tau}' for tau in tau_values])
    axes[1].set_xlabel('Bt (uM)')
    axes[1].set_ylabel('tau (ms)')
    axes[1].set_title(f'Mean spike count heatmap ({compartment_label})')
    fig.colorbar(im, ax=axes[1], label='mean spike count')

    plt.tight_layout()
    plt.show()


def main():
    mu = 0
    dt = 0.05
    T = 90000.0

    g_c = 0.09
    kappa = 0.9
    sigma = 1.0
    n_trials = 5

    y0 = default_two_compartment_initial_state()

    tau_values = [10, 100, 1000]
    Bt_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    soma_mean, soma_std, ais_mean, ais_std = run_tau_Bt_sweep_two_compartment(
        tau_values, Bt_values, sigma, g_c, kappa, T=T, dt=dt, mu=mu,
        n_trials=n_trials, y0=y0,
    )
    plot_tau_Bt_sweep(tau_values, Bt_values, soma_mean, soma_std, compartment_label="soma")
    plot_tau_Bt_sweep(tau_values, Bt_values, ais_mean, ais_std, compartment_label="AIS")


if __name__ == "__main__":
    main()