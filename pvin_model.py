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


def _safe_exp(x):
    """np.exp with the argument clipped to avoid overflow warnings/inf."""
    return np.exp(np.clip(x, -500.0, 500.0))


def _compartment_derivatives(V, h, n1, n3, Cai, r, m, Bt, gSK, ksk, gCa, gM):
    """Compute the ionic current sum and gating/calcium derivatives."""
    gNa, VNa = 300.0, 58.0
    gKv1, gKv3, VK = 15.0, 180.0, -80.0
    VCa = 68.0
    Vleak, gleak = -50.0, 8.0
    pgamma = 0.01

    Vm, Sm = -17.5, -11.4
    Aah, Sah, Vah = 0.0025, 10.0, 23.0
    Abh, Sbh, Vbh = 0.094, -5.5, -31.0

    Aan1, Van1, San1 = 0.0020, -30.0, -9.0
    Abn1, Vbn1, Sbn1 = 0.0170, -35.0, 5.9

    Aan3, Van3, San3 = 1.98, 96.0, -12.6
    Abn3, Vbn3, Sbn3 = 0.34, -36.0, 10.5

    Va, Sa = 3.0, -10.4
    nk = 5
    gh, Eh = 1.5, -30.0

    F = 0.0964853321
    mArea = 3000.0
    d = 0.1
    Car = 0.07
    KD = 0.1

    mmax = 1.0 / (1.0 + _safe_exp((V - Vm) / Sm))
    ah = Aah / _safe_exp((V - Vah) / Sah)
    bh = Abh * (V - Vbh) / (1.0 - _safe_exp((V - Vbh) / Sbh))
    INa = gNa * mmax**3 * h * (V - VNa)

    an1 = Aan1 * (V - Van1) / (1.0 - _safe_exp((V - Van1) / San1))
    bn1 = Abn1 / _safe_exp((V - Vbn1) / Sbn1)
    IKv1 = gKv1 * n1**4 * (V - VK)

    an3 = Aan3 * (V - Van3) / (1.0 - _safe_exp((V - Van3) / San3))
    bn3 = Abn3 / _safe_exp((V - Vbn3) / Sbn3)
    IKv3 = gKv3 * n3**2 * (V - VK)

    amax = 1.0 / (1.0 + _safe_exp((V - Va) / Sa))
    ICa = gCa * amax**2 * (V - VCa)

    k = Cai**nk / (ksk**nk + Cai**nk)
    ISK = gSK * k * (V - VK)

    Ileak = gleak * (V - Vleak)

    r_inf = 1.0 / (1.0 + _safe_exp((V + 84.0) / 10.2))
    tau_r = 1.0 / (_safe_exp(-14.59 - 0.086 * V) + _safe_exp(-1.87 + 0.0701 * V))
    tau_r = max(tau_r, 1e-6)
    Ih = gh * r * (V - Eh)

    m_inf = 1.0 / (1.0 + _safe_exp(-(V + 25.0) / 11.0))
    tau_m = 1.0 / (
        0.003 / _safe_exp(-(V + 78.0) / 19.0)
        + 0.003 / _safe_exp((V + 78.0) / 19.0)
    )
    tau_m = max(tau_m, 1e-6)
    IM = gM * m * (V - VK)

    I_ionic = Ileak + INa + IKv1 + IKv3 + ICa + ISK + Ih + IM

    dh = ah * (1.0 - h) - bh * h
    dn1 = an1 * (1.0 - n1) - bn1 * n1
    dn3 = an3 * (1.0 - n3) - bn3 * n3
    dCai = (-ICa / (2.0 * F * mArea * d) - pgamma * (Cai - Car)) / (1.0 + Bt / KD)
    dr = (r_inf - r) / tau_r
    dm = (m_inf - m) / tau_m

    return I_ionic, dh, dn1, dn3, dCai, dr, dm


def pvin_hh_two_compartment(t, y, Bt, Iapp, g_c, kappa,
                             gSK=10.0, ksk=0.8, gCa=8.0, gM=5.0,
                             gSK_AIS=10.0, ksk_AIS=0.8, gCa_AIS=8.0, gM_AIS=5.0,
                             Inoise=0.0, Cm=30.0):
    """Right-hand side of the two-compartment (soma + AIS) PVIN model."""
    (V, h, n1, n3, Cai, r, m,
     V_AIS, h_AIS, n1_AIS, n3_AIS, Cai_AIS, r_AIS, m_AIS) = y

    I_ionic, dh, dn1, dn3, dCai, dr, dm = _compartment_derivatives(
        V, h, n1, n3, Cai, r, m, Bt, gSK, ksk, gCa, gM
    )
    I_ionic_AIS, dh_AIS, dn1_AIS, dn3_AIS, dCai_AIS, dr_AIS, dm_AIS = _compartment_derivatives(
        V_AIS, h_AIS, n1_AIS, n3_AIS, Cai_AIS, r_AIS, m_AIS, Bt,
        gSK_AIS, ksk_AIS, gCa_AIS, gM_AIS
    )

    I_axial_soma = (g_c / kappa) * (V_AIS - V)
    I_axial_ais = (g_c / (1.0 - kappa)) * (V - V_AIS)

    dV = (-I_ionic + Iapp + I_axial_soma) / Cm + Inoise
    dV_AIS = (-I_ionic_AIS + I_axial_ais) / Cm

    return [dV, dh, dn1, dn3, dCai, dr, dm,
            dV_AIS, dh_AIS, dn1_AIS, dn3_AIS, dCai_AIS, dr_AIS, dm_AIS]


def run_pvin_two_compartment_with_ou(t_noise, I_OU, Bt, y0, g_c, kappa,
                                      gSK=10.0, ksk=0.8, gCa=8.0, gM=5.0,
                                      gSK_AIS=10.0, ksk_AIS=0.8, gCa_AIS=8.0,
                                      gM_AIS=5.0, Cm=30.0,
                                      rtol=1e-3, atol=1e-4):
    """Integrate the two-compartment PVIN model driven by OU noise."""

    def inoise_at(t):
        return np.interp(t, t_noise, I_OU)

    def rhs(t, y):
        Iapp = 0.0
        Inoise = inoise_at(t)
        return pvin_hh_two_compartment(
            t, y, Bt, Iapp, g_c, kappa,
            gSK=gSK, ksk=ksk, gCa=gCa, gM=gM,
            gSK_AIS=gSK_AIS, ksk_AIS=ksk_AIS, gCa_AIS=gCa_AIS, gM_AIS=gM_AIS,
            Inoise=Inoise, Cm=Cm,
        )

    sol = solve_ivp(
        rhs,
        t_span=(t_noise[0], t_noise[-1]),
        y0=y0,
        t_eval=t_noise,
        method="LSODA",
        rtol=rtol,
        atol=atol,
    )
    return sol


def generate_ou_noise(T, dt, mu, tau, sigma, seed=None):
    """Generate an Ornstein-Uhlenbeck noise trace via Euler-Maruyama."""
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


def count_spikes(t, V, threshold=-20.0, min_isi=2.0):
    """Count action potentials in a voltage trace via threshold crossing."""
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


def default_soma_initial_state():
    """Return the shared 7-state resting condition used across scripts."""
    return [
        -50.908283904949386,
        0.9874477040858244,
        0.017700897448150392,
        0.0017833075574000554,
        0.14045481906365542,
        0.037533345103523436,
        0.08664551707487766,
    ]


def default_two_compartment_initial_state():
    """Return the shared 14-state resting condition for soma and AIS."""
    soma_state = default_soma_initial_state()
    return soma_state + soma_state.copy()