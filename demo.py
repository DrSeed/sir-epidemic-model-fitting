# Self-contained demo: fit the first 20 days vs the whole epidemic.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sir import sir_incidence, likelihood_surface, r0_interval

BETA, GAMMA, REPORTING = 0.30, 0.10, 0.3


def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(9)
    truth = sir_incidence(BETA, GAMMA, days=120)
    cases = rng.poisson(REPORTING * truth)
    betas, gammas = np.linspace(0.15, 0.7, 70), np.linspace(0.03, 0.45, 70)

    rows, fits = [], {}
    for label, days in [("first 20 days", 20), ("full 120 days", 120)]:
        B, G, ll = likelihood_surface(cases[:days], betas, gammas, REPORTING)
        best_r0, lo, hi = r0_interval(B, G, ll)
        i = np.unravel_index(np.argmax(ll), ll.shape)
        fits[label] = (B, G, ll, B[i], G[i])
        rows.append({"data_used": label, "best_beta": round(B[i], 3), "best_gamma": round(G[i], 3),
                     "growth_rate_beta_minus_gamma": round(B[i] - G[i], 3),
                     "R0_best": round(best_r0, 2), "R0_low": round(lo, 2), "R0_high": round(hi, 2), "R0_true": BETA / GAMMA})
    table = pd.DataFrame(rows)
    table.to_csv("results/summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].bar(np.arange(120), cases, color="#cccccc", label="reported cases")
    for (label, (B, G, ll, b, g)), c in zip(fits.items(), ["#c0504d", "#2a7f62"]):
        axes[0].plot(REPORTING * sir_incidence(b, g, days=120), color=c, label=f"fit to {label} (R0={b / g:.1f})")
    axes[0].axvline(20, color="k", ls=":", lw=1)
    axes[0].set_xlabel("day")
    axes[0].set_ylabel("daily cases")
    axes[0].set_title("Same early data, very different futures")
    axes[0].legend(fontsize=8)
    for ax, (label, (B, G, ll, b, g)) in zip(axes[1:], fits.items()):
        rel = np.clip(ll - ll.max(), -40, 0)
        im = ax.contourf(B, G, rel, levels=20, cmap="viridis")
        ax.contour(B, G, rel, levels=[-1.92], colors="white", linewidths=1.5)
        ax.plot(BETA, GAMMA, "r*", ms=14, label="truth")
        ax.set_xlabel("beta (transmission)")
        ax.set_ylabel("gamma (recovery)")
        ax.set_title(f"Log-likelihood, {label}")
        ax.legend(loc="upper left")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig("figures/demo.png", dpi=120)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
