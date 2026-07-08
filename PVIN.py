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


def pvin_hh(t, y, Bt, Iapp, gSK=10.0, ksk=0.8, gCa=8.0):
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

    dV = (-Ileak - INa - IKv1 - IKv3 - ICa - ISK - Ih + Iapp) / Cm
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

    def iapp_at(t):
        return np.interp(t, t_noise, I_OU)

    def rhs(t, y):
        Iapp = iapp_at(t)
        return pvin_hh(t, y, Bt, Iapp, gSK=gSK, ksk=ksk, gCa=gCa)

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


def main():
    mu = 200.0
    tau = 3.0
    sigma = 50.0
    dt = 0.05
    T = 1000.0

    t_noise, I_OU = generate_ou_noise(T, dt, mu, tau, sigma, seed=0)

    y0 = [-69.9853, 0.99944, 0.000454, 0.000112, 0.02960, 0.25]
    Bt = 10.0

    sol = run_pvin_with_ou(t_noise, I_OU, Bt, y0)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    axes[0].plot(t_noise, I_OU, 'k', linewidth=0.5)
    axes[0].set_xlabel('T (ms)')
    axes[0].set_ylabel('I_OU (pA)')
    axes[0].set_title('OU noise input current')

    axes[1].plot(sol.t, sol.y[0], 'k', linewidth=0.5)
    axes[1].set_xlabel('T (ms)')
    axes[1].set_ylabel('V (mV)')
    axes[1].set_title(f'PVIN response to OU noise (Bt={Bt} uM)')
    axes[1].set_ylim(-80, 50)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()