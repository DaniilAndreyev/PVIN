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
                                 gSK, ksk, gCa, gM, gh, f0):
    """JIT-compiled ionic current sum and gating/calcium derivatives."""
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
    Eh = -30.0

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
def _pvin_hh_core_nb(y, Bt, Iapp, gSK, ksk, gCa, gM, gh, f0, Inoise):
    """JIT-compiled single-compartment RHS core. y is a length-8 array."""
    V, h, n1, n3, Cai, mh1, mh2, m = (
        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
    )
    Cm = 30.0

    I_ionic, dh, dn1, dn3, dCai, dmh1, dmh2, dm = _compartment_derivatives_nb(
        V, h, n1, n3, Cai, mh1, mh2, m, Bt, gSK, ksk, gCa, gM, gh, f0
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
                                      gSK, ksk, gCa, gM, gh, f0,
                                      gSK_AIS, ksk_AIS, gCa_AIS, gM_AIS, gh_AIS, f0_AIS,
                                      Inoise, Cm):
    """JIT-compiled two-compartment RHS core. y is a length-16 array."""
    V, h, n1, n3, Cai, mh1, mh2, m = (
        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
    )
    V_AIS, h_AIS, n1_AIS, n3_AIS, Cai_AIS, mh1_AIS, mh2_AIS, m_AIS = (
        y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]
    )

    I_ionic, dh, dn1, dn3, dCai, dmh1, dmh2, dm = _compartment_derivatives_nb(
        V, h, n1, n3, Cai, mh1, mh2, m, Bt, gSK, ksk, gCa, gM, gh, f0
    )
    I_ionic_AIS, dh_AIS, dn1_AIS, dn3_AIS, dCai_AIS, dmh1_AIS, dmh2_AIS, dm_AIS = _compartment_derivatives_nb(
        V_AIS, h_AIS, n1_AIS, n3_AIS, Cai_AIS, mh1_AIS, mh2_AIS, m_AIS, Bt,
        gSK_AIS, ksk_AIS, gCa_AIS, gM_AIS, gh_AIS, f0_AIS
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


def pvin_hh(t, y, Bt, Iapp, gSK=10.0, ksk=0.8, gCa=8.0, Inoise=0.0, gM=5.0,
            gh=10.0, f0=0.6):
    """Right-hand side of the single-compartment PVIN Hodgkin-Huxley system.

    Thin Python wrapper around the JIT-compiled core; same-style signature
    as before, now with gh (I_h conductance) and f0 (slow/fast I_h weight,
    Eq. 3) added.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    return _pvin_hh_core_nb(y_arr, Bt, Iapp, gSK, ksk, gCa, gM, gh, f0, Inoise)


def pvin_hh_two_compartment(t, y, Bt, Iapp, g_c, kappa,
                             gSK=10.0, ksk=0.8, gCa=8.0, gM=5.0, gh=10.0, f0=0.6,
                             gSK_AIS=10.0, ksk_AIS=0.8, gCa_AIS=8.0, gM_AIS=5.0,
                             gh_AIS=10.0, f0_AIS=0.6,
                             Inoise=0.0, Cm=30.0):
    """Right-hand side of the two-compartment (soma + AIS) PVIN model.

    Thin Python wrapper around the JIT-compiled core; same-style signature
    as before, now with gh/f0 (and AIS counterparts) added. NOTE: f0=0.6 is
    currently a placeholder pending the real value from the lab/paper.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    return _pvin_hh_two_compartment_core_nb(
        y_arr, Bt, Iapp, g_c, kappa,
        gSK, ksk, gCa, gM, gh, f0,
        gSK_AIS, ksk_AIS, gCa_AIS, gM_AIS, gh_AIS, f0_AIS,
        Inoise, Cm,
    )


def run_pvin_two_compartment_with_ou(t_noise, I_OU, Bt, y0, g_c, kappa,
                                      gSK=10.0, ksk=0.8, gCa=8.0, gM=5.0,
                                      gh=10.0, f0=0.6,
                                      gSK_AIS=10.0, ksk_AIS=0.8, gCa_AIS=8.0,
                                      gM_AIS=5.0, gh_AIS=10.0, f0_AIS=0.6, Cm=30.0,
                                      rtol=1e-2, atol=1e-3):
    """Integrate the two-compartment PVIN model driven by OU noise."""

    def inoise_at(t):
        return np.interp(t, t_noise, I_OU)

    def rhs(t, y):
        Iapp = 0.0
        Inoise = inoise_at(t)
        return pvin_hh_two_compartment(
            t, y, Bt, Iapp, g_c, kappa,
            gSK=gSK, ksk=ksk, gCa=gCa, gM=gM, gh=gh, f0=f0,
            gSK_AIS=gSK_AIS, ksk_AIS=ksk_AIS, gCa_AIS=gCa_AIS, gM_AIS=gM_AIS,
            gh_AIS=gh_AIS, f0_AIS=f0_AIS,
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


def run_pvin_with_ou(t_noise, I_OU, Bt, y0, gSK=10.0, ksk=0.8, gCa=8.0,
                      gM=1.0, gh=10.0, f0=0.6, rtol=1e-4, atol=1e-5):
    """Integrate the single-compartment PVIN model driven by OU noise."""

    def inoise_at(t):
        return np.interp(t, t_noise, I_OU)

    def rhs(t, y):
        Iapp = 0.0
        Inoise = inoise_at(t)
        return pvin_hh(t, y, Bt, Iapp, gSK=gSK, ksk=ksk, gCa=gCa,
                       Inoise=Inoise, gM=gM, gh=gh, f0=f0)

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
    """Return the shared 8-state resting condition used across scripts.

    State order: [V, h, n1, n3, Cai, mh1, mh2, m]
    Computed by settling the model (Bt=90, Iapp=0, gh=10, f0=0.6) to
    equilibrium under the two-component I_h formulation.
    """
    return [
        -47.39964981903278,
        0.9719528712565635,
        0.04061024120659064,
        0.003207495822362351,
        0.14575235157860392,
        0.17214638168093646,
        0.17453040765834063,
        0.1154408059181245,
    ]


def default_two_compartment_initial_state():
    """Return the shared 16-state resting condition for soma and AIS."""
    soma_state = default_soma_initial_state()
    return soma_state + soma_state.copy()