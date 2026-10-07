"""Run the experiment and save the figures.

1. Optimize land use for two objectives and plot the Pareto front.
2. Show three example maps: max revenue, knee point (balanced), max biodiversity.
3. Simple uncertainty test: perturb the yield map and see how the front shifts.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from landscape import make_landscape, CLASS_NAMES
from objectives import evaluate
from nsga2 import nsga2

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
LU_CMAP = ListedColormap(["#E3B23C", "#9BC53D", "#1B7340"])  # crop, pasture, natural


def knee_point(F):
    """Solution closest to the ideal point after normalizing both objectives."""
    norm = (F - F.min(0)) / (F.max(0) - F.min(0))
    return int(np.argmin(np.linalg.norm(1 - norm, axis=1)))


def plot_inputs(ls):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, key, title, cmap in zip(ax, ["yield_map", "habitat_map", "protected_mask"],
                                   ["Crop yield potential", "Habitat suitability",
                                    "Protected area"], ["YlOrBr", "Greens", "Greys"]):
        im = a.imshow(ls[key], cmap=cmap)
        a.set_title(title)
        a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "1_inputs.png"), dpi=150)


def plot_front(F, picks):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    order = np.argsort(F[:, 0])
    ax.plot(F[order, 0], F[order, 1], "o-", color="#3366AA", ms=4, lw=1,
            label="Pareto front (NSGA-II)")
    for (name, i), c in zip(picks.items(), ["#E3B23C", "#C0392B", "#1B7340"]):
        ax.plot(*F[i], "s", ms=10, color=c, label=name)
    ax.set_xlabel("Agricultural revenue (relative units)")
    ax.set_ylabel("Biodiversity score (relative units)")
    ax.set_title("Trade-off between agriculture and biodiversity")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "2_pareto_front.png"), dpi=150)


def plot_maps(pop, F, picks):
    fig, ax = plt.subplots(1, 3, figsize=(11, 4.8))
    for a, (name, i) in zip(ax, picks.items()):
        a.imshow(pop[i], cmap=LU_CMAP, vmin=0, vmax=2)
        a.set_title(f"{name}\nrevenue={F[i,0]:.0f}, biodiversity={F[i,1]:.0f}", fontsize=10)
        a.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=LU_CMAP(k)) for k in range(3)]
    fig.legend(handles, CLASS_NAMES.values(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(os.path.join(FIG, "3_example_maps.png"), dpi=150)


def uncertainty_experiment(ls, n_samples=5, noise=0.15, generations=150):
    """Re-run the optimization with noisy yield maps (simple Monte Carlo)."""
    rng = np.random.default_rng(123)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s in range(n_samples):
        noisy = dict(ls)
        noisy["yield_map"] = np.clip(ls["yield_map"] * (1 + rng.normal(0, noise, ls["yield_map"].shape)), 0, None)
        _, F = nsga2(noisy, evaluate, generations=generations, seed=s, verbose=False)
        o = np.argsort(F[:, 0])
        ax.plot(F[o, 0], F[o, 1], "-", alpha=0.7, label=f"yield sample {s + 1}")
        print(f"uncertainty sample {s + 1}/{n_samples} done")
    ax.set_xlabel("Agricultural revenue (relative units)")
    ax.set_ylabel("Biodiversity score (relative units)")
    ax.set_title(f"Pareto fronts under yield uncertainty (±{int(noise*100)}% noise)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "4_uncertainty.png"), dpi=150)


if __name__ == "__main__":
    ls = make_landscape(n=30, seed=42)
    plot_inputs(ls)
    pop, F = nsga2(ls, evaluate, pop_size=100, generations=200, seed=0)
    picks = {"Max revenue": int(np.argmax(F[:, 0])),
             "Knee point (balanced)": knee_point(F),
             "Max biodiversity": int(np.argmax(F[:, 1]))}
    plot_front(F, picks)
    plot_maps(pop, F, picks)
    if "--skip-uncertainty" not in sys.argv:
        uncertainty_experiment(ls)
    print("Figures saved in", os.path.abspath(FIG))
