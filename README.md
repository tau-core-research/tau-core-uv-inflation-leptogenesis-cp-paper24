# Paper 24: Scale-Invariant Simplicial $R^2$ Inflation, Multi-Sector $CP$ Structure, and Zero-Parameter Non-Thermal Scalaron Leptogenesis

**Author:** Jozsef Olcsak (ORCID: `0009-0003-7311-6522`)  
**Series:** Tau Core Paper 24 / Standalone Journal Package `Sub5` (*JCAP / PRD* format, `k_added = 0`)

## Overview
Investigates whether Starobinsky $R^2$ inflation, post-inflationary reheating, multi-sector $CP/CPT$ structure ($\bar{\theta}_{\mathrm{QCD}} = 0$, PMNS $\delta_{\mathrm{CP}} \approx -90^\circ$), and Type-I seesaw non-thermal leptogenesis are closed with zero added free parameters (`k_added = 0`) from the tetrahedral ($N=4$) simplicial-Clifford invariant $u = \sqrt{3}/24$ and chiral $\mathrm{Spin}(10)$ seesaw threshold $M_R = M_*/(256\pi^2) = 5.565\times 10^{14}\;\mathrm{GeV}$.

## Key Results
- **Theorem 2.2 (Minimal 1-Loop $a_2$ No-Go):** Massless Weyl and Yang-Mills fields give zero local $R^2$, and minimal scalars ($\xi=0, N_0\le 210$) fall short of $\beta_{R^2}^{\mathrm{Star}} = 4.839\times 10^8$ by $>3.34\times 10^9$.
- **Theorem 3.1 ($N=4$ Coleman-Weinberg Supertrace & $\sqrt{3}$ Cancellation):** Under classical UV scale invariance (`BC-UV-CW`), $n_{\mathrm{eff}}^{\mathrm{STr}}(N) = d_A - 2d_S$ equals $-1$ (tachyonic) at $N=3$ and $+2$ ($8-6$) at $N=4$, fixing $M_{\mathrm{inf}}/M_{\mathrm{Pl}} = \frac{5}{12{,}288\pi^3}$ ($M_{\mathrm{inf}} = 3.196\times 10^{13}\;\mathrm{GeV}$, $N_* = 51.41$, $A_s = 2.101\times 10^{-9}$, $n_s = 0.9620$, $r = 0.00426$).
- **Theorem 4.1 & Exact 4-ODE Boltzmann Closure:** $\mathbf{R}(\theta_\tau + iu) \in \mathrm{SO}(2,\mathbb{C})$ gives $|\varepsilon_1| = 8.470\times 10^{-5}$ and $\eta_B = +6.106\times 10^{-10}$ (`-0.35 sigma` vs. Planck 2018 for $28/79$ sphaleron; $5.585\times 10^{-10}$ for $12/37$ crossover).
- **ACT DR6 Four-Terminal Extension (Paper 28):** Coupling Jordan-frame Prescription II running ($\kappa_J = 0.01380$) to a stiff post-inflationary $O_T/O_Q$ kinetic phase ($w_{\mathrm{rh}} = 0.9015$, $\xi_H = 0$, $N_* = 59.72$) simultaneously hits ACT DR6 $n_s = 0.9740$ ($0.00\sigma$) and $A_s = 2.100\times 10^{-9}$ ($0.00\sigma$) at $r = 0.00502$ while preserving $\eta_B = 6.106\times 10^{-10}$ and passing BBN+ACT with $\Delta N_{\mathrm{eff}}^{\mathrm{GW}} = 8.41\times 10^{-6} \ll 0.17$ (see `tau-core-multiterminal-act-hubble-isw-paper28`).

## Reproducibility
```bash
python3 scripts/verify_paper24_starobinsky_seesaw_leptogenesis.py
python3 scripts/generate_paper24_figures.py
```
