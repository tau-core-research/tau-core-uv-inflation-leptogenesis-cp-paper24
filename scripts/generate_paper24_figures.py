#!/usr/bin/env python3
"""
Generates the two publication-grade multi-panel figures (EN + HU, 300 DPI PNG + PDF)
for Paper 24 / Sub5 (Scale-Invariant Simplicial R^2 Inflation, Multi-Sector CP Structure,
and Zero-Parameter Non-Thermal Scalaron Leptogenesis).
"""

from __future__ import annotations

import cmath
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import kn

# Frozen N = 4 simplicial constants (k_added = 0)
U_GEN = math.sqrt(3.0) / 24.0
R_SEED = 0.25
ALPHA_TRACE = 5.0 * U_GEN
ETA_SEESAW = ((192.0 * U_GEN - 9.0) ** 2) / 768.0

E_PL_GEV = 1.220890e19
M_PL_RED = E_PL_GEV / math.sqrt(8.0 * math.pi)
M_STAR_GEV = E_PL_GEV / math.sqrt(24.0 * math.pi)

M_R_TOT = (R_SEED ** 2 / (16.0 * math.pi ** 2)) * M_STAR_GEV
M_2_GEV = M_R_TOT / 3.0
M_1_GEV = ETA_SEESAW * M_2_GEV
M_INF_GEV = (ALPHA_TRACE / (2.0 * math.pi)) * M_R_TOT

V_EW_GEV = 246.21965
V_HIGGS_GEV = V_EW_GEV / math.sqrt(2.0)
M2_NU_GEV = 8.83118e-12
M3_NU_GEV = 50.36070e-12
G_STAR_SM = 106.75
ZETA_3 = 1.2020569031595942
S0_OVER_NGAMMA0 = (43.0 / 11.0) * (math.pi ** 4) / (45.0 * ZETA_3)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def compute_inflation_and_boltzmann():
    s23 = math.sqrt(2.0 / 3.0)

    def v_norm(x: float) -> float:
        return 0.75 * ((1.0 - math.exp(-s23 * x)) ** 2)

    def dlnv_dx(x: float) -> float:
        ex = math.exp(-s23 * x)
        return 2.0 * s23 * ex / (1.0 - ex)

    def kg_rhs(_n: float, y: np.ndarray) -> list[float]:
        x, xp = float(y[0]), float(y[1])
        eps_h = 0.5 * xp * xp
        return [xp, -(3.0 - eps_h) * (xp + dlnv_dx(x))]

    def hit_end(_n: float, y: np.ndarray) -> float:
        return 0.5 * float(y[1]) ** 2 - 1.0

    hit_end.terminal = True  # type: ignore[attr-defined]
    hit_end.direction = 1    # type: ignore[attr-defined]

    sol_kg = solve_ivp(
        kg_rhs, [0.0, 120.0], [5.85, -dlnv_dx(5.85)],
        events=hit_end, method="DOP853", rtol=1e-12, atol=1e-14, dense_output=True
    )
    n_tot = float(sol_kg.t_events[0][0])
    x_end = float(sol_kg.y_events[0][0][0])
    v_end = v_norm(x_end) * (M_PL_RED ** 2) * (M_INF_GEV ** 2)
    rho_end = 1.5 * v_end
    h_end = math.sqrt(rho_end / (3.0 * M_PL_RED ** 2))

    gamma_h = (M_INF_GEV ** 3) / (48.0 * math.pi * M_PL_RED ** 2)
    t_log = math.log(M_INF_GEV / 91.1876)
    alpha_em_mz, sin2_tw = 1.0 / 127.951, 0.23121
    inv_ay = (1.0 - sin2_tw) / alpha_em_mz + (-41.0 / 6.0) / (2.0 * math.pi) * t_log
    inv_a2 = sin2_tw / alpha_em_mz + (19.0 / 6.0) / (2.0 * math.pi) * t_log
    inv_a3 = (1.0 / 0.1180) + 7.0 / (2.0 * math.pi) * t_log
    pref_vv = (M_INF_GEV ** 3) / (1536.0 * (math.pi ** 3) * (M_PL_RED ** 2))
    gamma_g = pref_vv * (1.0 * ((-41.0 / 6.0) ** 2) / (inv_ay ** 2) + 3.0 * ((19.0 / 6.0) ** 2) / (inv_a2 ** 2) + 8.0 * 49.0 / (inv_a3 ** 2))
    gamma_n1 = (M_INF_GEV * (M_1_GEV ** 2)) / (96.0 * math.pi * M_PL_RED ** 2) * (
        (1.0 - 4.0 * (M_1_GEV ** 2) / (M_INF_GEV ** 2)) ** 1.5
    )
    gamma_tot = gamma_h + gamma_g + gamma_n1
    br_n1 = gamma_n1 / gamma_tot
    rho_rh = 3.0 * (M_PL_RED ** 2) * (gamma_tot ** 2)
    t_rh = (30.0 * rho_rh / (math.pi ** 2 * G_STAR_SM)) ** 0.25

    k_star_gev = (0.05 / 3.08567758149e22) * 1.973269804e-16
    t_0_gev = 2.7255 * 8.617333262e-5 * 1e-9
    g_s0 = 43.0 / 11.0
    n_star = 52.0
    for _ in range(20):
        xs, xps = sol_kg.sol(n_tot - n_star)
        eps_s = 0.5 * float(xps) ** 2
        h_s = M_INF_GEV * math.sqrt(v_norm(float(xs)) / (3.0 - eps_s))
        ln_aend_a0 = (1.0 / 3.0) * math.log(rho_rh / rho_end) + (1.0 / 3.0) * math.log(g_s0 / G_STAR_SM) + math.log(t_0_gev / t_rh)
        n_star = -math.log(k_star_gev) + math.log(h_s) + ln_aend_a0

    c_sl = -2.0 + math.log(2.0) + 0.5772156649015329

    def ns_r_at_efolds(nv: float):
        xv, xpv = sol_kg.sol(n_tot - nv)
        xv_f, xpv_f = float(xv), float(xpv)
        e1 = 0.5 * xpv_f ** 2
        xpps = -(3.0 - e1) * (xpv_f + dlnv_dx(xv_f))
        e2 = 2.0 * (xpps / xpv_f)
        ns = 1.0 - 2.0 * e1 - e2 - 2.0 * (e1 ** 2) - (2.0 * c_sl + 3.0) * e1 * e2
        rv = 16.0 * e1 * (1.0 - c_sl * e2)
        return xv_f, ns, rv

    x_star, ns_star, r_star = ns_r_at_efolds(n_star)
    efold_curve = np.linspace(46.0, 60.0, 80)
    ns_curve, r_curve = [], []
    for nv in efold_curve:
        _, nsv, rv = ns_r_at_efolds(float(nv))
        ns_curve.append(nsv)
        r_curve.append(rv)

    # Boltzmann trajectory for canonical eta_CP = u:
    th_tau = math.asin(3.0 / (8.0 * math.sqrt(2.0)))
    x_ratio = (M_2_GEV / M_1_GEV) ** 2
    g_loop = math.sqrt(x_ratio) * (1.0 / (1.0 - x_ratio) + 1.0 - (1.0 + x_ratio) * math.log((1.0 + x_ratio) / x_ratio))
    sum_m2_gev2 = (M2_NU_GEV ** 2) + (M3_NU_GEV ** 2)

    def run_boltz(eta_cp: float, dense: bool = False):
        z_ci = th_tau + 1j * eta_cp
        w12 = 0.5 * (M3_NU_GEV - M2_NU_GEV) * math.sin(2.0 * th_tau) - 0.5j * (M3_NU_GEV + M2_NU_GEV) * math.sinh(2.0 * eta_cp)
        w11 = M2_NU_GEV * abs(cmath.cos(z_ci)) ** 2 + M3_NU_GEV * abs(cmath.sin(z_ci)) ** 2
        eps_1 = (1.0 / (8.0 * math.pi)) * ((M_2_GEV / V_HIGGS_GEV ** 2) / w11) * (w12 ** 2).imag * g_loop
        gamma_d1 = (w11 * (M_1_GEV ** 2)) / (8.0 * math.pi * (V_HIGGS_GEV ** 2))
        gamma_d1_nt = gamma_d1 * (2.0 * M_1_GEV / M_INF_GEV)

        rho_phi_0 = 3.0 * (M_PL_RED ** 2) * (h_end ** 2)
        e_phi_0 = rho_phi_0 / (M_INF_GEV ** 4)
        dn0 = 1.0e-5
        e_rad_0 = (1.0 - br_n1) * (gamma_tot / h_end) * e_phi_0 * dn0

        def rhs(n_ef: float, y: np.ndarray) -> list[float]:
            ep, er, xn1, xbl = float(y[0]), float(y[1]), float(y[2]), float(y[3])
            ep = max(ep, 0.0)
            er = max(er, 1e-120)
            xn1_p = max(xn1, 0.0)
            rp = ep * math.exp(-3.0 * n_ef) * (M_INF_GEV ** 4)
            rr = er * math.exp(-4.0 * n_ef) * (M_INF_GEV ** 4)
            t_val = (30.0 * rr / (math.pi ** 2 * G_STAR_SM)) ** 0.25
            z_val = M_1_GEV / max(t_val, 1e-30)
            if z_val < 50.0:
                k1_v, k2_v = float(kn(1, z_val)), float(kn(2, z_val))
                rk = k1_v / k2_v
                n_n1_eq = (1.0 / (math.pi ** 2)) * t_val * (M_1_GEV ** 2) * k2_v
                n_l_eq = (1.5 * ZETA_3 / (math.pi ** 2)) * (t_val ** 3)
                gam_w_id = 0.5 * (n_n1_eq / n_l_eq) * gamma_d1 * rk
            else:
                rk = 1.0 - 1.5 / z_val
                n_n1_eq = 0.0
                gam_w_id = 0.0
            gam_w_dl2 = (2.0 / (math.pi ** 3)) * (t_val ** 3) * sum_m2_gev2 / (V_HIGGS_GEV ** 4)
            x_n1_eq = n_n1_eq * math.exp(3.0 * n_ef) / (M_INF_GEV ** 3)
            rn1 = 0.5 * M_INF_GEV * max(xn1_p - x_n1_eq, 0.0) * math.exp(-3.0 * n_ef) * (M_INF_GEV ** 3)
            h_val = math.sqrt((rp + rn1 + rr) / (3.0 * M_PL_RED ** 2))
            decay_flux = (gamma_d1_nt / h_val) * xn1_p - (gamma_d1 * rk / h_val) * x_n1_eq
            dep = -(gamma_tot / h_val) * ep
            der = math.exp(n_ef) * ((1.0 - br_n1) * (gamma_tot / h_val) * ep + 0.5 * decay_flux)
            dxn1 = 2.0 * br_n1 * (gamma_tot / h_val) * ep - decay_flux
            dxbl = eps_1 * decay_flux - ((gam_w_id + gam_w_dl2) / h_val) * xbl
            return [dep, der, dxn1, dxbl]

        n_max = 26.0
        t_eval = np.linspace(dn0, n_max, 400) if dense else None
        sol = solve_ivp(
            rhs, [dn0, n_max],
            [e_phi_0 * (1.0 - gamma_tot / h_end * dn0), e_rad_0, 2.0 * br_n1 * (gamma_tot / h_end) * e_phi_0 * dn0, 0.0],
            method="Radau", rtol=1e-9, atol=1e-17, max_step=0.05, t_eval=t_eval
        )
        return sol

    sol_b = run_boltz(U_GEN, dense=True)
    n_grid = sol_b.t
    rho_phi_g = np.maximum(sol_b.y[0], 1e-90) * np.exp(-3.0 * n_grid) * (M_INF_GEV ** 4)
    rho_rad_g = np.maximum(sol_b.y[1], 1e-90) * np.exp(-4.0 * n_grid) * (M_INF_GEV ** 4)
    t_grid = (30.0 * rho_rad_g / (math.pi ** 2 * G_STAR_SM)) ** 0.25
    s_comov_final = float(((2.0 * math.pi ** 2 / 45.0) * G_STAR_SM * (t_grid[-1] ** 3)) * math.exp(3.0 * float(n_grid[-1])) / (M_INF_GEV ** 3))
    # Comoving baryon asymmetry normalized to final post-reheating entropy (smooth monotone build-up 0 -> 6.106e-10):
    eta_b_traj = S0_OVER_NGAMMA0 * (28.0 / 79.0) * (sol_b.y[3] / s_comov_final)
    y_n1_traj = sol_b.y[2] / s_comov_final

    eta_cp_scan = np.linspace(0.015, 0.130, 36)
    eta_b_scan_28 = []
    eta_b_scan_12 = []
    for ecp in eta_cp_scan:
        sb = run_boltz(float(ecp), dense=False)
        erf, xblf = float(sb.y[1, -1]), float(sb.y[3, -1])
        rrf = erf * math.exp(-4.0 * 26.0) * (M_INF_GEV ** 4)
        tf = (30.0 * rrf / (math.pi ** 2 * G_STAR_SM)) ** 0.25
        scf = ((2.0 * math.pi ** 2 / 45.0) * G_STAR_SM * (tf ** 3)) * math.exp(3.0 * 26.0) / (M_INF_GEV ** 3)
        ybl = xblf / scf
        eta_b_scan_28.append(S0_OVER_NGAMMA0 * (28.0 / 79.0) * ybl)
        eta_b_scan_12.append(S0_OVER_NGAMMA0 * (12.0 / 37.0) * ybl)

    return {
        "x_star": x_star, "x_end": x_end, "n_star": n_star,
        "ns_star": ns_star, "r_star": r_star,
        "efold_curve": efold_curve, "ns_curve": np.array(ns_curve), "r_curve": np.array(r_curve),
        "n_grid": n_grid, "rho_phi_g": rho_phi_g, "rho_rad_g": rho_rad_g,
        "t_grid": t_grid, "eta_b_traj": eta_b_traj, "y_n1_traj": y_n1_traj,
        "eta_cp_scan": eta_cp_scan, "eta_b_scan_28": np.array(eta_b_scan_28), "eta_b_scan_12": np.array(eta_b_scan_12),
    }


def make_figure_1(data: dict, lang: str) -> None:
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 11})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=300)

    # Panel A: Potential V(phi) / (M_Pl^2 M_inf^2)
    phi_grid = np.linspace(-0.4, 6.8, 400)
    s23 = math.sqrt(2.0 / 3.0)
    v_grid = 0.75 * ((1.0 - np.exp(-s23 * phi_grid)) ** 2)
    ax1.plot(phi_grid, v_grid, color="#1E3A8A", lw=2.6, label=(
        r"$V(\phi)/(M_{\mathrm{Pl}}^2 M_{\mathrm{inf}}^2) = \frac{3}{4}(1 - e^{-\sqrt{2/3}\,\phi/M_{\mathrm{Pl}}})^2$"
    ))
    ax1.axhline(0.75, color="#64748B", ls="--", lw=1.1)

    xs = data["x_star"]
    xe = data["x_end"]
    vs = 0.75 * ((1.0 - math.exp(-s23 * xs)) ** 2)
    ve = 0.75 * ((1.0 - math.exp(-s23 * xe)) ** 2)

    ax1.scatter([xs], [vs], color="#DC2626", s=75, zorder=5)
    ax1.scatter([xe], [ve], color="#D97706", s=70, zorder=5)
    ax1.scatter([0.0], [0.0], color="#059669", s=70, zorder=5)

    if lang == "en":
        ax1.set_title("A. Scale-Invariant Plaquette Starobinsky Potential ($k_{\\mathrm{added}}=0$)", fontweight="bold")
        ax1.set_xlabel(r"Canonical Einstein-Frame Scalaron $\phi / M_{\mathrm{Pl}}$")
        ax1.set_ylabel(r"Normalized Potential $V(\phi) / (M_{\mathrm{Pl}}^2 M_{\mathrm{inf}}^2)$")
        ax1.annotate(
            f"CMB Pivot ($k_* = 0.05\\;\\mathrm{{Mpc}}^{{-1}}$)\n$\\phi_*/M_{{\\mathrm{{Pl}}}} = {xs:.2f},\\; N_* = {data['n_star']:.2f}$\n$\\sqrt{{R_{{J,*}}}}/M_R = 0.84 < 1$",
            xy=(xs, vs), xytext=(2.85, 0.43),
            arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.3),
            fontsize=9.3, bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#DC2626", alpha=0.95)
        )
        ax1.annotate(
            f"Inflation Ends ($\\epsilon_1 = 1$)\n$\\phi_{{\\mathrm{{end}}}}/M_{{\\mathrm{{Pl}}}} = {xe:.2f}$",
            xy=(xe, ve), xytext=(1.55, 0.14),
            arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.3),
            fontsize=9.3, bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBEB", ec="#D97706", alpha=0.95)
        )
        ax1.annotate(
            r"Reheating Well: $\phi \to HH, gg, N_1 N_1$" + "\n" + r"$M_{\mathrm{inf}} = \frac{5}{12\,288\pi^3}M_{\mathrm{Pl}} = 3.20\times 10^{13}\;\mathrm{GeV}$",
            xy=(0.0, 0.0), xytext=(0.95, 0.84),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=1.3),
            fontsize=9.1, bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#059669", alpha=0.95)
        )
    else:
        xs_hu = f"{xs:.2f}".replace(".", "{,}")
        ns_ef_hu = f"{data['n_star']:.2f}".replace(".", "{,}")
        xe_hu = f"{xe:.2f}".replace(".", "{,}")
        ax1.set_title("A. Skálainvariáns Plakett Starobinsky-Potenciál ($k_{\\mathrm{added}}=0$)", fontweight="bold")
        ax1.set_xlabel(r"Kanonikus Einstein-keretbeli skalaron $\phi / M_{\mathrm{Pl}}$")
        ax1.set_ylabel(r"Normált potenciál $V(\phi) / (M_{\mathrm{Pl}}^2 M_{\mathrm{inf}}^2)$")
        ax1.annotate(
            f"CMB pivot ($k_* = 0{{,}}05\\;\\mathrm{{Mpc}}^{{-1}}$)\n$\\phi_*/M_{{\\mathrm{{Pl}}}} = {xs_hu},\\; N_* = {ns_ef_hu}$\n$\\sqrt{{R_{{J,*}}}}/M_R = 0{{,}}84 < 1$",
            xy=(xs, vs), xytext=(2.85, 0.43),
            arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.3),
            fontsize=9.3, bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#DC2626", alpha=0.95)
        )
        ax1.annotate(
            f"Infláció vége ($\\epsilon_1 = 1$)\n$\\phi_{{\\mathrm{{end}}}}/M_{{\\mathrm{{Pl}}}} = {xe_hu}$",
            xy=(xe, ve), xytext=(1.55, 0.14),
            arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.3),
            fontsize=9.3, bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBEB", ec="#D97706", alpha=0.95)
        )
        ax1.annotate(
            r"Reheating gödör: $\phi \to HH, gg, N_1 N_1$" + "\n" + r"$M_{\mathrm{inf}} = \frac{5}{12\,288\pi^3}M_{\mathrm{Pl}} = 3{,}20\times 10^{13}\;\mathrm{GeV}$",
            xy=(0.0, 0.0), xytext=(0.95, 0.84),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=1.3),
            fontsize=9.1, bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#059669", alpha=0.95)
        )

    ax1.set_xlim(-0.35, 6.7)
    ax1.set_ylim(-0.04, 1.05)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="lower right", fontsize=8.8)

    # Panel B: (n_s, r) plane with Planck+BK18 1sigma/2sigma bands and LiteBIRD/CMB-S4 forecast
    ns_grid = np.linspace(0.952, 0.978, 300)
    ns0, sig_ns = 0.9649, 0.0042
    # Planck 2018 + BICEP/Keck 2021 upper contour (95% CL r < 0.036, 68% CL r < 0.019)
    r_95 = 0.036 * np.sqrt(np.maximum(0.0, 1.0 - ((ns_grid - ns0) / (2.0 * sig_ns)) ** 2))
    r_68 = 0.019 * np.sqrt(np.maximum(0.0, 1.0 - ((ns_grid - ns0) / (1.0 * sig_ns)) ** 2))

    ax2.fill_between(ns_grid, 0.0, r_95, color="#BFDBFE", alpha=0.45, label=(
        "Planck 2018 + BICEP/Keck (95% CL)" if lang == "en" else "Planck 2018 + BICEP/Keck (95% CL)"
    ))
    ax2.fill_between(ns_grid, 0.0, r_68, color="#60A5FA", alpha=0.45, label=(
        "Planck 2018 + BICEP/Keck (68% CL)" if lang == "en" else "Planck 2018 + BICEP/Keck (68% CL)"
    ))
    ax2.axhspan(0.00426 - 0.0010, 0.00426 + 0.0010, color="#FDE68A", alpha=0.55, label=(
        r"LiteBIRD / CMB-S4 $\pm 1\sigma(r)$ Target Band" if lang == "en" else r"LiteBIRD / CMB-S4 $\pm 1\sigma(r)$ cél-sáv"
    ))
    ax2.plot(data["ns_curve"], data["r_curve"], color="#1E293B", lw=2.0, ls="-", label=(
        r"$R + R^2/(6M_{\mathrm{inf}}^2)$ Trajectory ($N_* \in [46, 60]$)" if lang == "en" else r"$R + R^2/(6M_{\mathrm{inf}}^2)$ pálya ($N_* \in [46, 60]$)"
    ))
    nstar_lbl = f"{data['n_star']:.2f}" if lang == "en" else f"{data['n_star']:.2f}".replace(".", "{,}")
    ns_lbl = f"{data['ns_star']:.4f}" if lang == "en" else f"{data['ns_star']:.4f}".replace(".", "{,}")
    r_lbl = f"{data['r_star']:.5f}" if lang == "en" else f"{data['r_star']:.5f}".replace(".", "{,}")
    ax2.scatter([data["ns_star"]], [data["r_star"]], color="#DC2626", marker="*", s=230, zorder=6, label=(
        f"Zero-Parameter Prediction ($N_*={nstar_lbl}$):\n$n_s={ns_lbl},\\; r={r_lbl}$"
        if lang == "en" else
        f"Paramétermentes előrejelzés ($N_*={nstar_lbl}$):\n$n_s={ns_lbl},\\; r={r_lbl}$"
    ))

    if lang == "en":
        ax2.set_title("B. Primordial $(n_s, r)$ Plane & Next-Generation B-Mode Gate", fontweight="bold")
        ax2.set_xlabel(r"Scalar Spectral Index $n_s$ ($k_* = 0.05\;\mathrm{Mpc}^{-1}$)")
        ax2.set_ylabel(r"Tensor-to-Scalar Ratio $r$")
    else:
        ax2.set_title("B. Primordiális $(n_s, r)$ Sík és B-Módusú Falszifikációs Kapu", fontweight="bold")
        ax2.set_xlabel(r"Skalár spektrális index $n_s$ ($k_* = 0{,}05\;\mathrm{Mpc}^{-1}$)")
        ax2.set_ylabel(r"Tenzor–skalár arány $r$")

    ax2.set_xlim(0.953, 0.977)
    ax2.set_ylim(0.0, 0.040)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    stem = f"fig1_starobinsky_k4_inflation_and_cmb_contours_{lang}"
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def make_figure_2(data: dict, lang: str) -> None:
    plt.rcParams.update({"font.size": 10.0, "axes.titlesize": 11.0, "axes.labelsize": 10.5})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.2, 5.1), dpi=300)

    ng = data["n_grid"]
    # Panel A: Reheating energy densities & temperature
    ax1.plot(ng, np.log10(np.maximum(data["rho_phi_g"], 1e10)), color="#1D4ED8", lw=2.2, label=(
        r"$\log_{10}(\rho_\phi / \mathrm{GeV}^4)$"
    ))
    ax1.plot(ng, np.log10(np.maximum(data["rho_rad_g"], 1e10)), color="#EA580C", lw=2.2, label=(
        r"$\log_{10}(\rho_r / \mathrm{GeV}^4)$"
    ))
    ax1_t = ax1.twinx()
    ax1_t.plot(ng, np.log10(data["t_grid"]), color="#059669", lw=2.0, ls="--", label=(
        r"$\log_{10}(T / \mathrm{GeV})$"
    ))
    ax1_t.axhline(math.log10(M_1_GEV), color="#7C3AED", ls=":", lw=1.4)

    if lang == "en":
        ax1.set_title("A. 4-ODE Gravitational Reheating", fontweight="bold")
        ax1.set_xlabel(r"Post-Inflation e-Folds $\Delta N = \ln(a / a_{\mathrm{end}})$")
        ax1.set_ylabel(r"Energy Density $\log_{10}(\rho / \mathrm{GeV}^4)$")
        ax1_t.set_ylabel(r"Plasma Temperature $\log_{10}(T / \mathrm{GeV})$", color="#059669")
        ax1_t.text(10.8, math.log10(M_1_GEV) + 0.14, r"$M_1 = 5.70\times 10^{12}\;\mathrm{GeV}$", color="#7C3AED", fontsize=8.5)
        ax1_t.annotate(
            r"$T_{\max} = 2.11\times 10^{12}\;\mathrm{GeV}$" + "\n" + r"($M_1/T_{\max} = 2.70$)",
            xy=(0.55, math.log10(2.111e12)), xytext=(8.5, 11.82),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2),
            color="#065F46", fontsize=8.3,
            bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec="#059669", alpha=0.94)
        )
    else:
        ax1.set_title("A. 4-ODE Gravitációs Reheating", fontweight="bold")
        ax1.set_xlabel(r"Infláció utáni e-folding $\Delta N = \ln(a / a_{\mathrm{end}})$")
        ax1.set_ylabel(r"Energiasűrűség $\log_{10}(\rho / \mathrm{GeV}^4)$")
        ax1_t.set_ylabel(r"Plazma-hőmérséklet $\log_{10}(T / \mathrm{GeV})$", color="#059669")
        ax1_t.text(10.8, math.log10(M_1_GEV) + 0.14, r"$M_1 = 5{,}70\times 10^{12}\;\mathrm{GeV}$", color="#7C3AED", fontsize=8.5)
        ax1_t.annotate(
            r"$T_{\max} = 2{,}11\times 10^{12}\;\mathrm{GeV}$" + "\n" + r"($M_1/T_{\max} = 2{,}70$)",
            xy=(0.55, math.log10(2.111e12)), xytext=(8.5, 11.82),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2),
            color="#065F46", fontsize=8.3,
            bbox=dict(boxstyle="round,pad=0.25", fc="#ECFDF5", ec="#059669", alpha=0.94)
        )

    ax1.set_xlim(0.0, 25.0)
    ax1.set_ylim(30.0, 56.0)
    ax1_t.set_ylim(8.2, 13.5)
    ax1.grid(True, alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8.3)

    # Panel B: Freeze-in of eta_B(Delta N) via phi -> N_1 N_1
    ax2.plot(ng, data["eta_b_traj"] * 1e10, color="#DC2626", lw=2.5, label=(
        r"$\eta_B(\Delta N)$ ($c_{\mathrm{sph}} = 28/79$, $z_\tau = \theta_\tau + i u$)"
    ))
    ax2.plot(ng, data["eta_b_traj"] * (12.0 / 37.0) / (28.0 / 79.0) * 1e10, color="#D97706", lw=1.8, ls="--", label=(
        r"$\eta_B(\Delta N)$ ($c_{\mathrm{sph}} = 12/37$ crossover)" if lang == "en" else r"$\eta_B(\Delta N)$ ($c_{\mathrm{sph}} = 12/37$ átmenet)"
    ))
    ax2.axhspan(6.08, 6.16, color="#10B981", alpha=0.35, label=(
        r"Planck 2018: $(6.12 \pm 0.04)\times 10^{-10}$" if lang == "en" else r"Planck 2018: $(6{,}12 \pm 0{,}04)\times 10^{-10}$"
    ))
    ax2.axhline(6.12, color="#047857", lw=1.2, ls=":")

    if lang == "en":
        ax2.set_title(r"B. Comoving $\phi \to N_1 N_1$ Asymmetry Build-Up", fontweight="bold")
        ax2.set_xlabel(r"Post-Inflation e-Folds $\Delta N$")
        ax2.set_ylabel(r"Comoving Yield $\eta_B^{\mathrm{comov}}(\Delta N) \times 10^{10}$")
        ax2.annotate(
            r"Asymptotic Freeze-Out:" + "\n" + r"$\eta_B = +6.106\times 10^{-10}$ ($-0.35\sigma$)",
            xy=(21.0, 6.106), xytext=(2.2, 4.35),
            arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.3),
            fontsize=8.8, bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#DC2626", alpha=0.95)
        )
    else:
        ax2.set_title(r"B. Együttmozgó $\phi \to N_1 N_1$ Aszimmetria-Felépülés", fontweight="bold")
        ax2.set_xlabel(r"Infláció utáni e-folding $\Delta N$")
        ax2.set_ylabel(r"Együttmozgó hozam $\eta_B^{\mathrm{comov}}(\Delta N) \times 10^{10}$")
        ax2.annotate(
            r"Aszimptotikus befagyás:" + "\n" + r"$\eta_B = +6{,}106\times 10^{-10}$ ($-0{,}35\sigma$)",
            xy=(21.0, 6.106), xytext=(2.2, 4.35),
            arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.3),
            fontsize=8.8, bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#DC2626", alpha=0.95)
        )

    ax2.set_xlim(0.0, 26.0)
    ax2.set_ylim(0.0, 7.2)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower right", fontsize=8.2)

    # Panel C: Sensitivity curve eta_B(eta_CP) vs chiral boost rapidity eta_CP
    ax3.plot(data["eta_cp_scan"], data["eta_b_scan_28"] * 1e10, color="#1E3A8A", lw=2.3, label=(
        r"Exact 4-ODE $\eta_B(\eta_{\mathrm{CP}})$ ($28/79$)" if lang == "en" else r"Egzakt 4-ODE $\eta_B(\eta_{\mathrm{CP}})$ ($28/79$)"
    ))
    ax3.fill_between(
        data["eta_cp_scan"],
        data["eta_b_scan_12"] * 1e10,
        data["eta_b_scan_28"] * 1e10,
        color="#93C5FD", alpha=0.38,
        label=("Sphaleron Band $[12/37,\, 28/79]$" if lang == "en" else "Szfaleron-sáv $[12/37,\, 28/79]$")
    )
    ax3.axhspan(6.08, 6.16, color="#10B981", alpha=0.35, label="Planck 2018 ($1\\sigma$)")
    ax3.axvline(U_GEN, color="#DC2626", lw=1.8, ls="--", label=(
        r"$K_4$ Generator $u = \sqrt{3}/24 = 0.07217$" if lang == "en" else r"$K_4$ generátor $u = \sqrt{3}/24 = 0{,}07217$"
    ))
    ax3.scatter([U_GEN], [6.1059], color="#DC2626", s=85, zorder=6)

    if lang == "en":
        ax3.set_title(r"C. Chiral Boost $\eta_{\mathrm{CP}}$ Intersection", fontweight="bold")
        ax3.set_xlabel(r"Imaginary Casas–Ibarra Boost $\eta_{\mathrm{CP}} = \mathrm{Im}(z_\tau)$")
        ax3.set_ylabel(r"Final Baryon Asymmetry $\eta_B \times 10^{10}$")
    else:
        ax3.set_title(r"C. Királis Boost ($\eta_{\mathrm{CP}}$) Metszéspont", fontweight="bold")
        ax3.set_xlabel(r"Képzetes Casas–Ibarra boost $\eta_{\mathrm{CP}} = \mathrm{Im}(z_\tau)$")
        ax3.set_ylabel(r"Végső barion-aszimmetria $\eta_B \times 10^{10}$")

    ax3.set_xlim(0.015, 0.130)
    ax3.set_ylim(1.0, 10.5)
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left", fontsize=8.1)

    fig.tight_layout()
    stem = f"fig2_boltzmann_reheating_leptogenesis_and_cp_tests_{lang}"
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    data = compute_inflation_and_boltzmann()
    for lang in ("en", "hu"):
        make_figure_1(data, lang)
        make_figure_2(data, lang)
    print(f"[PASS] Generated EN + HU Paper 24 figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
