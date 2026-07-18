"""
Tau/sigma sweep analysis and plotting for the PVIN model.
"""

import numpy as np
import matplotlib.pyplot as plt

from pvin_model import count_spikes, generate_ou_noise, run_pvin_with_ou


def run_tau_sigma_sweep(tau_values, sigma_values, T=90000.0, dt=0.05,
                         mu=0.0, Bt=90.0, seed=0,
                         y0=None, spike_threshold=-20.0, min_isi=2.0):
    """Sweep OU noise tau and sigma values and count spikes for each combination."""
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


def plot_tau_sigma_sweep(tau_values, sigma_values, spike_counts):
    """Plot spike count curves by tau and a tau/sigma heatmap."""
    sigma_arr = np.asarray(sigma_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, tau in enumerate(tau_values):
        axes[0].plot(sigma_arr, spike_counts[i], marker='o', label=f'tau={tau} ms')

        if len(sigma_arr) > 1 and np.std(spike_counts[i]) > 0:
            corr = np.corrcoef(sigma_arr, spike_counts[i])[0, 1]
        else:
            corr = float('nan')
        print(f"tau={tau} ms: correlation(sigma, spike count) = {corr:.3f}")

    axes[0].set_xlabel('sigma (pA)')
    axes[0].set_ylabel('spike count')
    axes[0].set_title('Spike count vs sigma, by tau')
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
    axes[1].set_title('Spike count heatmap')
    fig.colorbar(im, ax=axes[1], label='spike count')

    plt.tight_layout()
    plt.show()


def main():
    mu = 0
    dt = 0.05
    T = 90000.0
    Bt = 90.0

    y0 = [-49.52776733833847,
          0.9827093136484906,
          0.024625396067553557,
          0.0022470251324288853,
          0.14110408589099907,
          0.03293826520477305,
          0.0971064287710309]

    tau_values = [10, 100, 1000]
    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    spike_counts = run_tau_sigma_sweep(tau_values, sigma_values, T=T, dt=dt,
                                        mu=mu, Bt=Bt, y0=y0)
    plot_tau_sigma_sweep(tau_values, sigma_values, spike_counts)


if __name__ == "__main__":
    main()