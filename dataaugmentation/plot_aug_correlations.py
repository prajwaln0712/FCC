import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
TABLE_DIR = os.path.join(ROOT, "tables", "augmentation")
PLOT_DIR = os.path.join(ROOT, "plots")
COLORS = {"inception": "#2a6fbb", "dino": "#d9534f"}
NAMES = {"inception": "Inception", "dino": "DINO"}


def analyse(table):
    df = pd.read_csv(os.path.join(TABLE_DIR, f"aug_{table}.csv"))
    y = df["acc_adapt"].to_numpy()
    stats = {}
    for key in ("inception", "dino"):
        x = df[key].to_numpy()
        rho, rho_p = spearmanr(x, y)
        r, r_p = pearsonr(x, y)
        stats[key] = dict(pearson_r=r, pearson_p=r_p, spearman_rho=rho, spearman_p=rho_p)
    return df, stats


def plot(table, df, stats):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    fig, ax = plt.subplots(figsize=(11, 8))
    y = df["acc_adapt"]
    handles = []
    for key in ("inception", "dino"):
        x = df[key]
        x_z = (x - x.mean()) / x.std()  # plotting only; r and rho are invariant to this
        sns.regplot(x=x_z, y=y, ax=ax, color=COLORS[key],
                    scatter_kws={"s": 70, "alpha": 0.8, "edgecolor": "white", "linewidths": 1.2},
                    line_kws={"linewidth": 2})
        s = stats[key]
        handles.append(Line2D([], [], marker="o", linestyle="", color=COLORS[key], markersize=9,
                              label=NAMES[key]))
        handles.append(Line2D([], [], linestyle="",
                              label=f"{NAMES[key]}: $\\rho$={s['spearman_rho']:.2f} "
                                    f"(p={s['spearman_p']:.3f}), $r$={s['pearson_r']:.2f} "
                                    f"(p={s['pearson_p']:.3f})"))
    ax.margins(x=0.07, y=0.08)
    ax.legend(handles=handles, loc="best", fontsize=12, framealpha=0.9)
    ax.set_xlabel(f"FCC {table} (z-score)", fontsize=15)
    ax.set_ylabel("cifar10_acc_adapt", fontsize=16)
    ax.set_title(f"aug {table}", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"aug_{table}_correlation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    rows = []
    for table in ("mean", "ratio"):
        df, stats = analyse(table)
        plot(table, df, stats)
        for key, s in stats.items():
            rows.append({"table": table, "feature": key, **s})
            print(f"{table:5s} {key:9s} spearman rho={s['spearman_rho']:.3f} (p={s['spearman_p']:.3f})  "
                  f"pearson r={s['pearson_r']:.3f} (p={s['pearson_p']:.3f})")
    out = os.path.join(TABLE_DIR, "aug_correlations.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table", "feature", "pearson_r", "pearson_p",
                                          "spearman_rho", "spearman_p"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")
