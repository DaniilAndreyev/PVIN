"""
g_h vs g_Na sweep (soma only, deterministic — sigma=0, no OU noise, no
applied current). Tests whether I_h alone, or in combination with
elevated Na+ conductance, is sufficient to destabilize the resting state
into spontaneous spiking.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from pvin_model import count_spikes, default_soma_initial_state, pvin_hh


def run_gh_gNa_sweep(gh_values, gNa_values, Bt=92.0, gM=5.0, T=90000.0, dt=0.05,
                      y0=None, spike_threshold=-20.0, min_isi=2.0, rtol=1e-2, atol=1e-3):
    """
    Sweep g_h and g_Na with no noise/applied current, starting from the
    model's resting state, and count spontaneous spikes at each
    combination.

    Parameters
    ----------
    gh_values : sequence of float
        I_h conductances to test (nS).
    gNa_values : sequence of float
        Na+ conductances to test (nS).
    Bt : float, optional
        Calcium buffer capacity (uM).
    gM : float, optional
        M-current conductance, held fixed (nS).
    T : float, optional
        Simulation duration (ms).
    dt : float, optional
        Time resolution for spike detection (ms).
    y0 : array_like, shape (8,), optional
        Initial condition. Defaults to the model's resting state.
    spike_threshold, min_isi : see count_spikes.
    rtol, atol : solver tolerances.

    Returns
    -------
    spike_counts : ndarray, shape (len(gh_values), len(gNa_values))
        Number of spontaneous spikes for each (gh, gNa) combination.
    """
    if y0 is None:
        y0 = default_soma_initial_state()

    t_eval = np.linspace(0, T, int(T / dt) + 1)
    spike_counts = np.zeros((len(gh_values), len(gNa_values)), dtype=int)

    for i, gh in enumerate(gh_values):
        for j, gNa in enumerate(gNa_values):
            def rhs(t, y, gh=gh, gNa=gNa):
                return pvin_hh(t, y, Bt, 0.0, gM=gM, gh=gh, gNa=gNa)

            sol = solve_ivp(rhs, (0, T), y0, method='LSODA', rtol=rtol, atol=atol, t_eval=t_eval)
            n_spikes = count_spikes(sol.t, sol.y[0], threshold=spike_threshold, min_isi=min_isi)
            spike_counts[i, j] = n_spikes
            print(f"gh={gh:>5.1f} nS, gNa={gNa:>6.1f} nS -> {n_spikes} spikes")

    return spike_counts


def plot_gh_gNa_heatmap(gh_values, gNa_values, spike_counts):
    """Heatmap of spontaneous spike count over (gh, gNa)."""
    gh_arr = np.asarray(gh_values)
    gNa_arr = np.asarray(gNa_values)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        spike_counts,
        aspect='auto',
        origin='lower',
        extent=[gNa_arr[0], gNa_arr[-1], gh_arr[0], gh_arr[-1]],
    )
    ax.set_xlabel('g_Na (nS)')
    ax.set_ylabel('g_h (nS)')
    ax.set_title('Spontaneous spike count (soma, no noise)')
    fig.colorbar(im, ax=ax, label='spike count')

    plt.tight_layout()
    plt.show()


def main():
    Bt = 90.0
    gh_values = list(np.arange(0, 21, 2))       # 0..20, step 2
    gNa_values = list(np.arange(100, 201, 20))  # 100..200, bigger step (20)

    y0 = default_soma_initial_state()

    t0 = time.time()
    spike_counts = run_gh_gNa_sweep(gh_values, gNa_values, Bt=Bt, y0=y0)
    print(f"Total time: {time.time()-t0:.1f}s")

    plot_gh_gNa_heatmap(gh_values, gNa_values, spike_counts)


if __name__ == "__main__":
    main()