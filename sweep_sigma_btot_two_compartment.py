"""
Sigma vs Btot (calcium buffer capacity) sweep for the two-compartment
(soma + AIS) PVIN model, holding tau fixed.

Each (sigma, Bt) combination is run over multiple independent OU noise
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


def run_sigma_Bt_sweep_two_compartment(sigma_values, Bt_values, tau, g_c, kappa,
                                        T=90000.0, dt=0.05, mu=0.0,
                                        n_trials=5, seed0=0, y0=None,
                                        spike_threshold=-20.0, min_isi=2.0):
    """
    Sweep OU noise sigma and calcium buffer capacity (Bt) for the
    two-compartment model, at a fixed tau, averaging soma spike counts
    over n_trials independent noise realizations at each combination.

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
    T, dt, mu : see run_tau_sigma_sweep_two_compartment.
    n_trials : int, optional
        Number of independent noise realizations (seeds) averaged at each
        (sigma, Bt) combination.
    seed0 : int, optional
        Base random seed; trial i uses seed = seed0 + i.
    y0 : array_like, shape (16,)
        Initial condition [soma states..., AIS states...]. Required.
    spike_threshold, min_isi : see run_tau_sigma_sweep_two_compartment.

    Returns
    -------
    soma_mean : ndarray, shape (len(sigma_values), len(Bt_values))
        Mean soma spike count across trials, for each (sigma, Bt).
    soma_std : ndarray, same shape
        Std. dev. of soma spike count across trials.
    """

    if y0 is None:
        y0 = default_two_compartment_initial_state()

    soma_mean = np.zeros((len(sigma_values), len(Bt_values)))
    soma_std = np.zeros((len(sigma_values), len(Bt_values)))

    for i, sigma in enumerate(sigma_values):
        for j, Bt in enumerate(Bt_values):
            soma_trials = np.zeros(n_trials)
            for k in range(n_trials):
                t_noise, I_OU = generate_ou_noise(T, dt, mu, tau, sigma, seed=seed0 + k)
                sol = run_pvin_two_compartment_with_ou(t_noise, I_OU, Bt, y0, g_c, kappa)
                soma_trials[k] = count_spikes(
                    sol.t, sol.y[0], threshold=spike_threshold, min_isi=min_isi
                )

            soma_mean[i, j] = soma_trials.mean()
            soma_std[i, j] = soma_trials.std()
            print(f"sigma={sigma:.2f} pA, Bt={Bt:.1f} uM -> "
                  f"soma: {soma_mean[i, j]:.1f}+/-{soma_std[i, j]:.1f} spikes "
                  f"(n={n_trials})")

    return soma_mean, soma_std


def plot_sigma_Bt_sweep(sigma_values, Bt_values, spike_counts_mean):
    """Heatmap of mean soma spike count over (sigma, Bt)."""
    Bt_arr = np.asarray(Bt_values)

    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        spike_counts_mean,
        aspect='auto',
        origin='lower',
        extent=[Bt_arr[0], Bt_arr[-1], 0, len(sigma_values)],
    )
    ax.set_yticks(np.arange(len(sigma_values)) + 0.5)
    ax.set_yticklabels([f'{sigma}' for sigma in sigma_values])
    ax.set_xlabel('Bt (uM)')
    ax.set_ylabel('sigma (pA)')
    ax.set_title('Mean spike count heatmap')
    fig.colorbar(im, ax=ax, label='mean spike count')

    plt.tight_layout()
    plt.show()


def main():
    mu = 0
    dt = 0.05
    T = 90000.0

    g_c = 0.09
    kappa = 0.9
    tau = 100
    n_trials = 5

    y0 = default_two_compartment_initial_state()

    sigma_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    Bt_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    soma_mean, soma_std = run_sigma_Bt_sweep_two_compartment(
        sigma_values, Bt_values, tau, g_c, kappa, T=T, dt=dt, mu=mu,
        n_trials=n_trials, y0=y0,
    )
    plot_sigma_Bt_sweep(sigma_values, Bt_values, soma_mean)


if __name__ == "__main__":
    main()