import os

import matplotlib.pyplot as plt
import pandas as pd

CSV = "results/wyniki_symulacji.csv"
OUT_DIR = "results/figures"
WORLDS = ["baza", "jak WIG20", "jak S&P 500", "skew-t (baza)"]
METHODS = ["parametric", "residual", "bayesian"]
MODES = {"theoretical": "-", "empirical": "--"}          # styl linii wg kwantyla
COLORS = {"parametric": "tab:blue", "residual": "tab:orange", "bayesian": "tab:green"}
N_TICKS = [250, 500, 1000, 2500, 4000]

df = pd.read_csv(CSV, sep=";", decimal=",")
df = df[df["code_version"] == "nowy"]    # tylko nowy kod: 3 metody, 2 kwantyle, szerokosci


def draw_lines(ax, data, column):
    """Szesc linii: kolor = metoda, styl = kwantyl (ciagla teoretyczny, przerywana empiryczny)."""
    for method in METHODS:
        for mode, style in MODES.items():
            s = data[(data["method"] == method) & (data["quantile"] == mode)].sort_values("N")
            ax.plot(s["N"], s[column], style, marker="o", markersize=4,
                    color=COLORS[method], label=f"{method}, {mode}")


def format_n_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(N_TICKS)
    ax.set_xticklabels(N_TICKS)
    ax.minorticks_off()
    ax.grid(alpha=0.3)


def plot_coverage():
    fig, axes = plt.subplots(len(WORLDS), 2, figsize=(11, 12), sharey="row")
    for i, world in enumerate(WORLDS):
        for j, measure in enumerate(["VAR", "ES"]):
            ax = axes[i, j]
            data = df[(df["world"] == world) & (df["measure"] == measure)]
            draw_lines(ax, data, "coverage")
            ax.axhline(0.95, color="black", linestyle=":", linewidth=1)   # poziom nominalny
            format_n_axis(ax)
            ax.set_title(f"{world} - {measure}", fontsize=10)
            if j == 0:
                ax.set_ylabel("coverage")
            if i == len(WORLDS) - 1:
                ax.set_xlabel("N")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{OUT_DIR}/coverage_vs_N.png", dpi=150)


def plot_width():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for j, measure in enumerate(["VAR", "ES"]):
        ax = axes[j]
        data = df[(df["world"] == "baza") & (df["measure"] == measure)]
        draw_lines(ax, data, "mean_rel_width")

        # linia odniesienia ~ 1/sqrt(N), zaczepiona w punkcie N=4000 (bootstrap parametryczny, kwantyl teoretyczny)
        base = data[(data["method"] == "parametric") & (data["quantile"] == "theoretical")
                    & (data["N"] == 4000)]["mean_rel_width"].iloc[0]
        ns = sorted(data["N"].unique())
        ax.plot(ns, [base * (4000 / n) ** 0.5 for n in ns], color="gray", linestyle=":",
                label="~ 1/sqrt(N)")

        ax.set_yscale("log")
        format_n_axis(ax)
        ax.set_yticks(N_TICKS)                 # os log o waskim zakresie nie ma domyslnych podpisow
        ax.set_yticklabels(N_TICKS)
        ax.set_ylim(0.09, 1.1)
        ax.set_title(f"baza - {measure}", fontsize=10)
        ax.set_xlabel("N")
        if j == 0:
            ax.set_ylabel("mean relative width of the 95% CI")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"{OUT_DIR}/rel_width_vs_N.png", dpi=150)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_coverage()
    plot_width()
    print(f"Zapisano: {OUT_DIR}/coverage_vs_N.png oraz {OUT_DIR}/rel_width_vs_N.png")