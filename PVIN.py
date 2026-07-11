"""
PVIN (parvalbumin-expressing interneuron) Hodgkin-Huxley model with
Ornstein-Uhlenbeck noise input.

Reference:
    Ma, X., Miraucourt, L., Qiu, H., Sharif-Naeini, R., Khadra, A. (2023).
    Calcium buffering tunes intrinsic excitability of spinal dorsal horn
    parvalbumin-expressing interneurons: A computational model.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def pvin_hh(t, y, Bt, Iapp, gSK=10.0, ksk=0.8, gCa=8.0, Inoise=0.0):
    """
    Right-hand side of the PVIN Hodgkin-Huxley system.

    Parameters
    ----------
    t : float
        Time (ms).
    y : array_like, shape (6,)
        State vector [V, h, n1, n3, Cai, r]:
            V   - membrane voltage (mV)
            h   - Na+ inactivation gate
            n1  - Kv1 activation gate
            n3  - Kv3 activation gate
            Cai - intracellular calcium concentration (uM)
            r   - HCN activation gate
    Bt : float
        Calcium buffer capacity (uM).
    Iapp : float
        Applied/injected current at time t (pA).
    gSK : float, optional
        SK channel conductance (nS).
    ksk : float, optional
        SK channel calcium sensitivity (uM).
    gCa : float, optional
        Calcium channel conductance (nS).
    Inoise : float, optional
        Additional noise current term added directly to dV (not scaled by Cm).

    Returns
    -------
    list of float
        Derivatives [dV, dh, dn1, dn3, dCai, dr].
    """
    V, h, n1, n3, Cai, r = y

    # Fixed membrane parameters
    gNa, VNa = 300.0, 58.0
    gKv1, gKv3, VK = 15.0, 180.0, -80.0
    VCa = 68.0
    Vleak, gleak = -50.0, 8.0
    Cm = 30.0
    pgamma = 0.01

    # INa kinetics
    Vm, Sm = -17.5, -11.4
    Aah, Sah, Vah = 0.0025, 10.0, 23.0
    Abh, Sbh, Vbh = 0.094, -5.5, -31.0

    # IKv1 kinetics
    Aan1, Van1, San1 = 0.0020, -30.0, -9.0
    Abn1, Vbn1, Sbn1 = 0.0170, -35.0, 5.9

    # IKv3 kinetics
    Aan3, Van3, San3 = 1.98, 96.0, -12.6
    Abn3, Vbn3, Sbn3 = 0.34, -36.0, 10.5

    # ICa kinetics
    Va, Sa = 3.0, -10.4

    # ISK
    nk = 5

    # Ih (HCN)
    gh, Eh = 1.5, -30.0

    # Calcium dynamics
    F = 0.0964853321
    mArea = 3000.0
    d = 0.1
    Car = 0.07
    KD = 0.1

    # Sodium current
    mmax = 1.0 / (1.0 + np.exp((V - Vm) / Sm))
    ah = Aah / np.exp((V - Vah) / Sah)
    bh = Abh * (V - Vbh) / (1.0 - np.exp((V - Vbh) / Sbh))
    INa = gNa * mmax**3 * h * (V - VNa)

    # Kv1 current
    an1 = Aan1 * (V - Van1) / (1.0 - np.exp((V - Van1) / San1))
    bn1 = Abn1 / np.exp((V - Vbn1) / Sbn1)
    IKv1 = gKv1 * n1**4 * (V - VK)

    # Kv3 current
    an3 = Aan3 * (V - Van3) / (1.0 - np.exp((V - Van3) / San3))
    bn3 = Abn3 / np.exp((V - Vbn3) / Sbn3)
    IKv3 = gKv3 * n3**2 * (V - VK)

    # Calcium current
    amax = 1.0 / (1.0 + np.exp((V - Va) / Sa))
    ICa = gCa * amax**2 * (V - VCa)

    # SK current
    k = Cai**nk / (ksk**nk + Cai**nk)
    ISK = gSK * k * (V - VK)

    # Leak current
    Ileak = gleak * (V - Vleak)

    # HCN current
    r_inf = 1.0 / (1.0 + np.exp((V + 84.0) / 10.2))
    tau_r = 1.0 / (np.exp(-14.59 - 0.086 * V) + np.exp(-1.87 + 0.0701 * V))
    Ih = gh * r * (V - Eh)

    dV = (-Ileak - INa - IKv1 - IKv3 - ICa - ISK - Ih + Iapp) / Cm + Inoise
    dh = ah * (1.0 - h) - bh * h
    dn1 = an1 * (1.0 - n1) - bn1 * n1
    dn3 = an3 * (1.0 - n3) - bn3 * n3
    dCai = (-ICa / (2.0 * F * mArea * d) - pgamma * (Cai - Car)) / (1.0 + Bt / KD)
    dr = (r_inf - r) / tau_r

    return [dV, dh, dn1, dn3, dCai, dr]


def generate_ou_noise(T, dt, mu, tau, sigma, seed=None):
    """
    Generate an Ornstein-Uhlenbeck noise trace via Euler-Maruyama integration.

    Parameters
    ----------
    T : float
        Total duration (ms).
    dt : float
        Integration timestep (ms).
    mu : float
        Mean current (pA).
    tau : float
        Correlation time constant (ms).
    sigma : float
        Noise amplitude (pA).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    t_noise : ndarray
        Time vector (ms).
    I_OU : ndarray
        Noise current trace (pA).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt)) + 1
    t_noise = np.linspace(0, T, n_steps)
    I_OU = np.zeros(n_steps)
    I_OU[0] = mu

    for i in range(1, n_steps):
        I_OU[i] = (
            I_OU[i - 1]
            + dt * (-(I_OU[i - 1] - mu) / tau)
            + sigma * np.sqrt(2 * dt / tau) * rng.standard_normal()
        )

    return t_noise, I_OU


def run_pvin_with_ou(t_noise, I_OU, Bt, y0, gSK=10.0, ksk=0.8, gCa=8.0,
                      rtol=1e-4, atol=1e-5):
    """
    Integrate the PVIN model driven by a precomputed OU noise trace.

    Parameters
    ----------
    t_noise : ndarray
        Time points of the noise trace (ms).
    I_OU : ndarray
        Noise current values (pA).
    Bt : float
        Calcium buffer capacity (uM).
    y0 : array_like, shape (6,)
        Initial conditions [V, h, n1, n3, Cai, r].
    gSK, ksk, gCa : float, optional
        Model conductance/sensitivity overrides.
    rtol, atol : float, optional
        Solver tolerances.

    Returns
    -------
    scipy.integrate.OdeResult
        Solution object with attributes `.t` (time) and `.y` (state trajectories).
    """

    def inoise_at(t):
        return np.interp(t, t_noise, I_OU)

    def rhs(t, y):
        Iapp = 0.0
        Inoise = inoise_at(t)
        return pvin_hh(t, y, Bt, Iapp, gSK=gSK, ksk=ksk, gCa=gCa, Inoise=Inoise)

    sol = solve_ivp(
        rhs,
        t_span=(t_noise[0], t_noise[-1]),
        y0=y0,
        t_eval=t_noise,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    return sol


def count_spikes(t, V, threshold=-20.0, min_isi=2.0):
    """
    Count action potentials in a voltage trace via threshold crossing.
 
    A spike is counted each time V crosses `threshold` from below to above,
    with a refractory-style minimum inter-spike interval to avoid double
    counting noisy fluctuations near threshold.
 
    Parameters
    ----------
    t : ndarray
        Time vector (ms).
    V : ndarray
        Membrane voltage trace (mV).
    threshold : float, optional
        Voltage threshold (mV) defining a spike crossing.
    min_isi : float, optional
        Minimum time (ms) between counted spikes.
 
    Returns
    -------
    int
        Number of spikes detected.
    ndarray
        Times (ms) at which spikes were detected.
    """

    above = V > threshold
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]
    if crossings.size == 0:
        return 0

    spike_times = t[crossings]

    filtered = [spike_times[0]]
    for st in spike_times[1:]:
        if st - filtered[-1] >= min_isi:
            filtered.append(st)

    return len(filtered)


def run_tau_sigma_sweep(tau_values, sigma_values, T=90000.0, dt=0.05,
                         mu=0.0, Bt=90.0, seed=0,
                         y0=None, spike_threshold=-20.0, min_isi=2.0):
    """
    Sweep OU noise tau and sigma values and count spikes for each combination.
 
    Parameters
    ----------
    tau_values : sequence of float
        OU correlation time constants to test (ms).
    sigma_values : sequence of float
        OU noise amplitudes to test (pA).
    T : float, optional
        Simulation duration (ms).
    dt : float, optional
        OU noise integration timestep (ms).
    mu : float, optional
        OU mean current (pA).
    Bt : float, optional
        Calcium buffer capacity (uM).
    seed : int, optional
        Random seed (kept fixed across the sweep so differences in spike
        count reflect tau/sigma, not the noise realization).
    y0 : array_like, optional
        Initial condition. Defaults to a resting-state vector.
    spike_threshold : float, optional
        Voltage threshold used for spike counting (mV).
    min_isi : float, optional
        Minimum inter-spike interval used for spike counting (ms).
 
    Returns
    -------
    spike_counts : ndarray, shape (len(tau_values), len(sigma_values))
        Number of spikes for each (tau, sigma) combination.
    """


    spike_counts = np.zeros((len(tau_values), len(sigma_values)), dtype=int)

    for i, tau in enumerate(tau_values):
        for j, sigma in enumerate(sigma_values):
            t_noise, I_OU = generate_ou_noise(T, dt, mu, tau, sigma, seed=seed)
            sol = run_pvin_with_ou(t_noise, I_OU, Bt, y0)
            n_spikes = count_spikes(sol.t, sol.y[0],
                                        threshold=spike_threshold,
                                        min_isi=min_isi)
            spike_counts[i, j] = n_spikes
            print(f"tau={tau:>6.1f} ms, sigma={sigma:.2f} pA -> {n_spikes} spikes")

    return spike_counts


def plot_tau_sigma_sweep(tau_values, sigma_values, spike_counts):
    """
    Plot spike count as a function of sigma for each tau, plus a heatmap,
    and report the correlation between sigma and spike count for each tau.
 
    Parameters
    ----------
    tau_values : sequence of float
        OU correlation time constants used in the sweep (ms).
    sigma_values : sequence of float
        OU noise amplitudes used in the sweep (pA).
    spike_counts : ndarray, shape (len(tau_values), len(sigma_values))
        Spike counts from run_tau_sigma_sweep.
    """

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

    im = axes[1].imshow(spike_counts, aspect='auto', origin='lower',
                         extent=[sigma_arr[0], sigma_arr[-1], 0, len(tau_values)])
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
    tau = 1000
    sigma = 1
    dt = 0.05
    T = 90000.0
    print_time = 3000.0

    t_noise, I_noise = generate_ou_noise(T, dt, mu, tau, sigma, seed=0)

    y0 = [-49.086653, 0.980895, 0.027342, 0.002419, 0.141284, 0.031588]
    Bt = 90.0

    sol = run_pvin_with_ou(t_noise, I_noise, Bt, y0)

    values_at_print_time = [np.interp(print_time, sol.t, sol.y[i]) for i in range(6)]
    V, h, n1, n3, Cai, r = values_at_print_time
    print(f"At t={print_time:.1f} ms (3.0 s):")
    print(f"V={V:.6f}, h={h:.6f}, n1={n1:.6f}, n3={n3:.6f}, Cai={Cai:.6f}, r={r:.6f}")

    n_spikes = count_spikes(sol.t, sol.y[0])
    print(f"Total spikes detected: {n_spikes}")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    axes[0].plot(t_noise, I_noise, 'k', linewidth=0.5)
    axes[0].set_xlabel('T (ms)')
    axes[0].set_ylabel('I_noise (pA)')
    axes[0].set_title('OU noise input current')

    axes[1].plot(sol.t, sol.y[0], 'k', linewidth=0.5)
    axes[1].set_xlabel('T (ms)')
    axes[1].set_ylabel('V (mV)')
    axes[1].set_title(f'PVIN response to OU noise (Bt={Bt} uM)')
    axes[1].set_ylim(-80, 50)

    plt.tight_layout()
    plt.show()

    tau_values = [10, 100, 1000]
    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    spike_counts = run_tau_sigma_sweep(tau_values, sigma_values, T=T, dt=dt,
                                        mu=mu, Bt=Bt, y0=y0)
    plot_tau_sigma_sweep(tau_values, sigma_values, spike_counts)


if __name__ == "__main__":
    main()