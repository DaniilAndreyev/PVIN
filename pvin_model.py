"""
PVIN (parvalbumin-expressing interneuron) Hodgkin-Huxley model with
Ornstein-Uhlenbeck noise input.

The core channel/derivative math is JIT-compiled with Numba for speed.
Requires: pip install numba

I_h (HCN current) is modeled with two gating components (slow m_h1, fast
m_h2), weighted by fraction f0:
    I_h = g_h * (f0 * m_h1 + (1 - f0) * m_h2) * (V - E_h)

State vector order (single compartment, 8 states):
    [V, h, n1, n3, Cai, mh1, mh2, m]
State vector order (two compartment, 16 states):
    [soma states (8)..., AIS states (8)...]  -- AIS block starts at index 8

Reference:
    Ma, X., Miraucourt, L., Qiu, H., Sharif-Naeini, R., Khadra, A. (2023).
    Calcium buffering tunes intrinsic excitability of spinal dorsal horn
    parvalbumin-expressing interneurons: A computational model.
"""

import numpy as np
from scipy.integrate import solve_ivp
from numba import njit


def _safe_exp(x):
    """np.exp with the argument clipped to avoid overflow warnings/inf.

    Plain-numpy version, kept for any array-valued/interactive use outside
    the JIT-compiled hot path below.
    """
    return np.exp(np.clip(x, -500.0, 500.0))


@njit(cache=True)
def _safe_exp_nb(x):
    """Scalar, JIT-compiled version of _safe_exp used inside the hot path."""
    if x > 500.0:
        x = 500.0
    elif x < -500.0:
        x = -500.0
    return np.exp(x)


@njit(cache=True)
def _compartment_derivatives_nb(V, h, n1, n3, Cai, mh1, mh2, m, Bt,
                                 gSK, ksk, gCa, gM, gh, f0, gNa, gleak, Vleak):
    """JIT-compiled ionic current sum and gating/calcium derivatives."""
    VNa = 49.0
    gKv1, gKv3, VK = 20.0, 80.0, -88.0
    VCa = 63.0
    pgamma = 0.01

    # INa: m_inf = 1/(1+exp(-(V+25)/12.3))
    #   alpha_h = 0.0024/exp((V-5.0)/10.7)
    #   beta_h  = 0.060*(V+23)/(1-exp(-(V+23)/5.0))
    Vm, Sm = -25.0, -12.3
    Aah, Sah, Vah = 0.0024, 10.7, 5.0
    Abh, Sbh, Vbh = 0.060, -5.0, -23.0

    # IKv1.3: alpha_n1 = 0.002*(V+14)/(1-exp(-(V+14)/9.3))
    #                   beta_n1  = 0.014/exp((V+54)/6.0)
    Aan1, Van1, San1 = 0.0020, -14.0, -9.3
    Abn1, Vbn1, Sbn1 = 0.0140, -54.0, 6.0

    # IKv3.1: alpha_n3 = 1.8*(V-80)/(1-exp(-(V-80)/12))
    #         beta_n3  = 0.095/exp((V+36)/9.5)
    Aan3, Van3, San3 = 1.8, 80.0, -12.0
    Abn3, Vbn3, Sbn3 = 0.095, -36.0, 9.5

    # ICa: a_inf = 1/(1+exp(-(V+2.8)/5.0))
    Va, Sa = -2.8, -5.0
    nk = 5
    Eh = -22.0

    F = 0.0964853321
    mArea = 3000.0
    d = 0.1
    Car = 0.07
    KD = 0.1

    mmax = 1.0 / (1.0 + _safe_exp_nb((V - Vm) / Sm))
    ah = Aah / _safe_exp_nb((V - Vah) / Sah)
    bh = Abh * (V - Vbh) / (1.0 - _safe_exp_nb((V - Vbh) / Sbh))
    INa = gNa * mmax**3 * h * (V - VNa)

    an1 = Aan1 * (V - Van1) / (1.0 - _safe_exp_nb((V - Van1) / San1))
    bn1 = Abn1 / _safe_exp_nb((V - Vbn1) / Sbn1)
    IKv1 = gKv1 * n1**4 * (V - VK)

    an3 = Aan3 * (V - Van3) / (1.0 - _safe_exp_nb((V - Van3) / San3))
    bn3 = Abn3 / _safe_exp_nb((V - Vbn3) / Sbn3)
    IKv3 = gKv3 * n3**2 * (V - VK)

    amax = 1.0 / (1.0 + _safe_exp_nb((V - Va) / Sa))
    ICa = gCa * amax**2 * (V - VCa)

    k = Cai**nk / (ksk**nk + Cai**nk)
    ISK = gSK * k * (V - VK)

    Ileak = gleak * (V - Vleak)

    alpha1 = (-0.00292 * V - 0.445) / (1.0 - _safe_exp_nb((V + 152.397) / 24.22))
    beta1 = (0.0280 * V - 1.074) / (1.0 - _safe_exp_nb(-(V - 38.357) / 17.8))
    tau_h1 = 1.0 / (alpha1 + beta1)
    if tau_h1 < 1e-6:
        tau_h1 = 1e-6
    mh1_inf = alpha1 / (alpha1 + beta1)

    alpha2 = (-0.00318 * V - 0.700) / (1.0 - _safe_exp_nb((V + 220.126) / 26.0))
    beta2 = (0.0216 * V - 1.065) / (1.0 - _safe_exp_nb(-(V - 49.305) / 15.05))
    tau_h2 = 1.0 / (alpha2 + beta2)
    if tau_h2 < 1e-6:
        tau_h2 = 1e-6
    mh2_inf = alpha2 / (alpha2 + beta2)

    Ih = gh * (f0 * mh1 + (1.0 - f0) * mh2) * (V - Eh)

    m_inf = 1.0 / (1.0 + _safe_exp_nb(-(V + 25.0) / 11.0))
    tau_m = 1.0 / (
        0.003 / _safe_exp_nb(-(V + 78.0) / 19.0)
        + 0.003 / _safe_exp_nb((V + 78.0) / 19.0)
    )
    if tau_m < 1e-6:
        tau_m = 1e-6
    IM = gM * m * (V - VK)

    I_ionic = Ileak + INa + IKv1 + IKv3 + ICa + ISK + Ih + IM

    dh = ah * (1.0 - h) - bh * h
    dn1 = an1 * (1.0 - n1) - bn1 * n1
    dn3 = an3 * (1.0 - n3) - bn3 * n3
    dCai = (-ICa / (2.0 * F * mArea * d) - pgamma * (Cai - Car)) / (1.0 + Bt / KD)
    dmh1 = (mh1_inf - mh1) / tau_h1
    dmh2 = (mh2_inf - mh2) / tau_h2
    dm = (m_inf - m) / tau_m

    return I_ionic, dh, dn1, dn3, dCai, dmh1, dmh2, dm


@njit(cache=True)
def _pvin_hh_core_nb(y, Bt, Iapp, gSK, ksk, gCa, gM, gh, f0, gNa, gleak, Vleak, Inoise):
    """JIT-compiled single-compartment RHS core. y is a length-8 array."""
    V, h, n1, n3, Cai, mh1, mh2, m = (
        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
    )
    Cm = 30.0

    I_ionic, dh, dn1, dn3, dCai, dmh1, dmh2, dm = _compartment_derivatives_nb(
        V, h, n1, n3, Cai, mh1, mh2, m, Bt, gSK, ksk, gCa, gM, gh, f0, gNa, gleak, Vleak
    )
    dV = (-I_ionic + Iapp) / Cm + Inoise

    out = np.empty(8)
    out[0] = dV
    out[1] = dh
    out[2] = dn1
    out[3] = dn3
    out[4] = dCai
    out[5] = dmh1
    out[6] = dmh2
    out[7] = dm
    return out


@njit(cache=True)
def _pvin_hh_two_compartment_core_nb(y, Bt, Iapp, g_c, kappa,
                                      gSK, ksk, gCa, gM, gh, f0, gNa, gleak, Vleak,
                                      gSK_AIS, ksk_AIS, gCa_AIS, gM_AIS, gh_AIS, f0_AIS, gNa_AIS,
                                      gleak_AIS, Vleak_AIS,
                                      Inoise, Cm):
    """JIT-compiled two-compartment RHS core. y is a length-16 array."""
    V, h, n1, n3, Cai, mh1, mh2, m = (
        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
    )
    V_AIS, h_AIS, n1_AIS, n3_AIS, Cai_AIS, mh1_AIS, mh2_AIS, m_AIS = (
        y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]
    )

    I_ionic, dh, dn1, dn3, dCai, dmh1, dmh2, dm = _compartment_derivatives_nb(
        V, h, n1, n3, Cai, mh1, mh2, m, Bt, gSK, ksk, gCa, gM, gh, f0, gNa, gleak, Vleak
    )
    I_ionic_AIS, dh_AIS, dn1_AIS, dn3_AIS, dCai_AIS, dmh1_AIS, dmh2_AIS, dm_AIS = _compartment_derivatives_nb(
        V_AIS, h_AIS, n1_AIS, n3_AIS, Cai_AIS, mh1_AIS, mh2_AIS, m_AIS, Bt,
        gSK_AIS, ksk_AIS, gCa_AIS, gM_AIS, gh_AIS, f0_AIS, gNa_AIS, gleak_AIS, Vleak_AIS
    )

    I_axial_soma = (g_c / kappa) * (V_AIS - V)
    I_axial_ais = (g_c / (1.0 - kappa)) * (V - V_AIS)

    dV = (-I_ionic + Iapp + I_axial_soma) / Cm + Inoise
    dV_AIS = (-I_ionic_AIS + I_axial_ais) / Cm

    out = np.empty(16)
    out[0] = dV
    out[1] = dh
    out[2] = dn1
    out[3] = dn3
    out[4] = dCai
    out[5] = dmh1
    out[6] = dmh2
    out[7] = dm
    out[8] = dV_AIS
    out[9] = dh_AIS
    out[10] = dn1_AIS
    out[11] = dn3_AIS
    out[12] = dCai_AIS
    out[13] = dmh1_AIS
    out[14] = dmh2_AIS
    out[15] = dm_AIS
    return out


def pvin_hh(t, y, Bt, Iapp, gSK=10.0, ksk=0.8, gCa=6.0, Inoise=0.0, gM=5.0,
            gh=10.0, f0=0.6, gNa=108.0, gleak=4.0, Vleak=-60.0):
    """Right-hand side of the single-compartment PVIN Hodgkin-Huxley system."""
    y_arr = np.asarray(y, dtype=np.float64)
    return _pvin_hh_core_nb(y_arr, Bt, Iapp, gSK, ksk, gCa, gM, gh, f0, gNa, gleak, Vleak, Inoise)


def pvin_hh_two_compartment(t, y, Bt, Iapp, g_c, kappa,
                             gSK=10.0, ksk=0.8, gCa=6.0, gM=5.0, gh=10.0, f0=0.6, gNa=108.0,
                             gleak=4.0, Vleak=-60.0,
                             gSK_AIS=10.0, ksk_AIS=0.8, gCa_AIS=6.0, gM_AIS=5.0,
                             gh_AIS=10.0, f0_AIS=0.6, gNa_AIS=108.0,
                             gleak_AIS=4.0, Vleak_AIS=-60.0,
                             Inoise=0.0, Cm=30.0):
    """Right-hand side of the two-compartment (soma + AIS) PVIN model."""
    y_arr = np.asarray(y, dtype=np.float64)
    return _pvin_hh_two_compartment_core_nb(
        y_arr, Bt, Iapp, g_c, kappa,
        gSK, ksk, gCa, gM, gh, f0, gNa, gleak, Vleak,
        gSK_AIS, ksk_AIS, gCa_AIS, gM_AIS, gh_AIS, f0_AIS, gNa_AIS,
        gleak_AIS, Vleak_AIS,
        Inoise, Cm,
    )


def run_pvin_two_compartment_with_ou(t_noise, I_OU, Bt, y0, g_c, kappa,
                                      gSK=10.0, ksk=0.8, gCa=6.0, gM=5.0,
                                      gh=10.0, f0=0.6, gNa=108.0,
                                      gSK_AIS=10.0, ksk_AIS=0.8, gCa_AIS=6.0,
                                      gM_AIS=5.0, gh_AIS=10.0, f0_AIS=0.6, gNa_AIS=108.0,
                                      gleak_AIS=4.0, Vleak_AIS=-60.0, Cm=30.0,
                                      rtol=1e-2, atol=1e-3):
    """Integrate the two-compartment PVIN model driven by OU noise."""

    def inoise_at(t):
        return np.interp(t, t_noise, I_OU)

    def rhs(t, y):
        Iapp = 0.0
        Inoise = inoise_at(t)
        return pvin_hh_two_compartment(
            t, y, Bt, Iapp, g_c, kappa,
            gSK=gSK, ksk=ksk, gCa=gCa, gM=gM, gh=gh, f0=f0, gNa=gNa,
            gSK_AIS=gSK_AIS, ksk_AIS=ksk_AIS, gCa_AIS=gCa_AIS, gM_AIS=gM_AIS,
            gh_AIS=gh_AIS, f0_AIS=f0_AIS, gNa_AIS=gNa_AIS,
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


def run_pvin_with_ou(t_noise, I_OU, Bt, y0, gSK=10.0, ksk=0.8, gCa=6.0,
                      gM=5.0, gh=10.0, f0=0.6, gNa=108.0, rtol=1e-4, atol=1e-5):
    """Integrate the single-compartment PVIN model driven by OU noise."""

    def inoise_at(t):
        return np.interp(t, t_noise, I_OU)

    def rhs(t, y):
        Iapp = 0.0
        Inoise = inoise_at(t)
        return pvin_hh(t, y, Bt, Iapp, gSK=gSK, ksk=ksk, gCa=gCa,
                       Inoise=Inoise, gM=gM, gh=gh, f0=f0, gNa=gNa)

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
            + sigma * rng.standard_normal()
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
    """Return the shared 8-state resting condition used across scripts.

    State order: [V, h, n1, n3, Cai, mh1, mh2, m]
    Settled to equilibrium using the Table 1 parameter set and the
    paper's gating kinetics (Btot=92, Iapp=0).
    """
    return [
        -48.35074102597328,
        0.9733567207030432,
        0.24302314288668211,
        0.014781038574447984,
        0.10281834879704889,
        0.18287887541082554,
        0.18705868104079276,
        0.10690094928676495,
    ]


def default_two_compartment_initial_state():
    """Return the shared 16-state resting condition for soma and AIS."""
    soma_state = default_soma_initial_state()
    return soma_state + soma_state.copy()