"""
g_h vs g_M sweep (soma only, deterministic — sigma=0, no OU noise, no
applied current). Tests whether I_h alone, or in combination with the
M-current, is sufficient to destabilize the resting state into
spontaneous spiking.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from pvin_model import count_spikes, default_soma_initial_state, pvin_hh


def run_gh_gM_sweep(gh_values, gM_values, Bt=92.0, gNa=300.0, T=90000.0, dt=0.05,
                     y0=None, spike_threshold=-20.0, min_isi=2.0, rtol=1e-2, atol=1e-3):
    """
    Sweep g_h and g_M with no noise/applied current, starting from the
    model's resting state, and count spontaneous spikes at each
    combination.

    Parameters
    ----------
    gh_values : sequence of float
        I_h conductances to test (nS).
    gM_values : sequence of float
        M-current conductances to test (nS).
    Bt : float, optional
        Calcium buffer capacity (uM).
    gNa : float, optional
        Na+ conductance, held fixed (nS).
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
    spike_counts : ndarray, shape (len(gh_values), len(gM_values))
        Number of spontaneous spikes for each (gh, gM) combination.
    """
    if y0 is None:
        y0 = default_soma_initial_state()

    t_eval = np.linspace(0, T, int(T / dt) + 1)
    spike_counts = np.zeros((len(gh_values), len(gM_values)), dtype=int)

    for i, gh in enumerate(gh_values):
        for j, gM in enumerate(gM_values):
            def rhs(t, y, gh=gh, gM=gM):
                return pvin_hh(t, y, Bt, 0.0, gM=gM, gh=gh, gNa=gNa)

            sol = solve_ivp(rhs, (0, T), y0, method='LSODA', rtol=rtol, atol=atol, t_eval=t_eval)
            n_spikes = count_spikes(sol.t, sol.y[0], threshold=spike_threshold, min_isi=min_isi)
            spike_counts[i, j] = n_spikes
            print(f"gh={gh:>5.1f} nS, gM={gM:>5.1f} nS -> {n_spikes} spikes")

    return spike_counts


def plot_gh_gM_heatmap(gh_values, gM_values, spike_counts):
    """Heatmap of spontaneous spike count over (gh, gM)."""
    gh_arr = np.asarray(gh_values)
    gM_arr = np.asarray(gM_values)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        spike_counts,
        aspect='auto',
        origin='lower',
        extent=[gM_arr[0], gM_arr[-1], gh_arr[0], gh_arr[-1]],
    )
    ax.set_xlabel('g_M (nS)')
    ax.set_ylabel('g_h (nS)')
    ax.set_title('Spontaneous spike count (soma, no noise)')
    fig.colorbar(im, ax=ax, label='spike count')

    plt.tight_layout()
    plt.show()


def main():
    Bt = 90.0
    gh_values = list(np.arange(0, 21, 2))    # 0..20, step 2
    gM_values = list(np.arange(0, 16, 1))    # 0..15, step 1

    y0 = default_soma_initial_state()

    t0 = time.time()
    spike_counts = run_gh_gM_sweep(gh_values, gM_values, Bt=Bt, y0=y0)
    print(f"Total time: {time.time()-t0:.1f}s")

    plot_gh_gM_heatmap(gh_values, gM_values, spike_counts)


if __name__ == "__main__":
    main()