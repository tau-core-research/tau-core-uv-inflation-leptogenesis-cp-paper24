# Paper 24: Scale-Invariant Simplicial $R^2$ Inflation, Multi-Sector $CP$ Structure, and Zero-Parameter Non-Thermal Scalaron Leptogenesis

**Author:** Jozsef Olcsak (ORCID: `0009-0003-7311-6522`)  
**Series:** Tau Core Paper 24 / Standalone Journal Package `Sub5` (*JCAP / PRD* format, `v1.2.0`, `k_added = 0`)  
**Archive DOI:** `10.5281/zenodo.24380371` (Concept DOI: `10.5281/zenodo.24379737`)

## Overview
Investigates whether Starobinsky $R^2$ inflation, post-inflationary reheating, multi-sector $CP/CPT$ structure ($\bar{\theta}_{\mathrm{QCD}}^{\mathrm{tree}} = 0$, PMNS $\delta_{\mathrm{CP}} \approx -90^\circ$), and Type-I seesaw non-thermal leptogenesis are closed with zero added free parameters (`k_added = 0`) from the tetrahedral ($N=4$) simplicial-Clifford invariant $u = \sqrt{3}/24$ and chiral $\mathrm{Spin}(10)$ seesaw threshold $M_R = M_*/(256\pi^2) = 5.565\times 10^{14}\;\mathrm{GeV}$ ($M_* = M_{\mathrm{Pl}}/\sqrt{3}$).

## Key Results (`v1.2.0`)
- **Theorem 2.2 (Minimal 1-Loop $a_2$ No-Go):** Symbolic `SymPy` Lichnerowicz and Feynman-gauge Faddeev–Popov heat kernels confirm that massless Weyl ($s=1/2$) and Yang–Mills ($s=1$) fields contribute identically zero local $R^2$, while minimal real scalars ($\xi=0, N_0\le 210$) fall short of $\beta_{R^2}^{\mathrm{Star}} = 4.839\times 10^8$ by $>3.34\times 10^9$.
- **Theorem 3.1 ($N=4$ Coleman–Weinberg Supertrace, $\sqrt{3}$ Cancellation & Plaquette $R^3$ Stability):** Under Einstein-frame scale-invariant renormalization (`BC-UV-CW`, Prescription I), $n_{\mathrm{eff}}^{\mathrm{STr}}(N) = d_A - 2d_S$ equals $-1$ (tachyonic) at $N=3$ and $+2$ ($8-6$) at $N=4$, with unit Weyl-induced vacuum kinetic normalization $Z_{\mathrm{Weyl}}(0) = 3M_*^2/M_{\mathrm{Pl}}^2 \equiv 1$ at $Z_\Phi^{\mathrm{bare}} = 0$, sub-plaquette excursion $\Phi_* = 0.2909\,M_* = 0.1679\,M_{\mathrm{Pl}}$, plaquette cubic suppression $|\Delta\mathcal{L}_{R^3}/\mathcal{L}_{R^2}|_* = 2.283\times 10^{-16}\,|c_3|$, and $M_{\mathrm{inf}}/M_{\mathrm{Pl}} = \frac{5}{12{,}288\pi^3}$ ($M_{\mathrm{inf}} = 3.196\times 10^{13}\;\mathrm{GeV}$, $N_* = 51.412$, $A_s = 2.1011\times 10^{-9}$ [`+0.04 sigma`], $n_s = 0.9620$ [`-0.69 sigma` Planck 2018], $r = 0.00426$).
- **Theorem 4.1, Flavored Spectator Closure & Non-Minimal Higgs $\xi_H$ Pinning:** $\mathbf{R}(\theta_\tau + \mathrm{i}u) \in \mathrm{SO}(2,\mathbb{C})$ fixes the signed asymmetry $\varepsilon_1^{(B-L)} = +8.4705\times 10^{-5} > 0$ (net matter without $|\cdot|$), yielding $\eta_B = +6.106\times 10^{-10}$ (`-0.35 sigma` vs. Planck 2018 for $28/79$ sphaleron; $+5.587\times 10^{-10}$ for $12/37$ crossover) in the energy-conserving 4-ODE system and $+6.108\times 10^{-10}$ (`+0.027%` / `+0.028%` shift) across both the `SymPy`-derived 2-flavor $(\alpha_\perp, \tau)$ ($T \in [10^9, 10^{11}]\;\mathrm{GeV}$) and 3-flavor $(e,\mu,\tau)$ ($T \le 10^9\;\mathrm{GeV}$) SM chemical-potential spectator matrices $C^{(\ell+H)}$. Conformal SM Higgs coupling $\xi_H = 1/6$ boosts $\mathrm{BR}(\phi \to N_1 N_1)$ to $93.25\%$ and overproduces $\eta_B = +52.19\times 10^{-10}$ (`+1,151.7 sigma` excluded), pinning $\xi_H \in [-0.00184, +0.00256]$ at $2\sigma$.
- **ACT DR6 Companion Audit (Paper 28 `v4.3.0`):** Establishes a Three-Branch Prescription II No-Go (`k_added = 0`: fermion-only hilltop collapse $\kappa_J = -0.1953$, $K_4$ threshold multiplet $10^9 A_s = 1.349$ [`-25.05 sigma`], and gauged $U(1)_{B-L}$ $10^9 A_s = 5.675$ [`+119.18 sigma`]), while in the two-field $(\phi, \psi_{\mathrm{kin}})$ kination extension (`k_{UV,eff} = 1`), COBE normalization $10^9 A_s \equiv 2.100$ and 5-ODE non-thermal leptogenesis enforce an upper ceiling $n_s \le 0.9729$ ($2\sigma$) with minimax concordance at $n_s = 0.9722$, $\eta_B = +6.095\times 10^{-10}$, and $\Delta N_{\mathrm{eff}}^{\mathrm{GW}} = 1.39\times 10^{-8} \ll 0.17$.

## Reproducibility
```bash
python3 scripts/verify_paper24_starobinsky_seesaw_leptogenesis.py
python3 scripts/generate_paper24_figures.py
```
