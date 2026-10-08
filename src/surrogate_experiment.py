"""Experiment 1: a learned surrogate for an "expensive" indicator.

In real projects an indicator such as biodiversity can come from a slow
model (e.g. species distribution models run on every candidate map).
NSGA-II needs tens of thousands of evaluations, so this becomes the
bottleneck. Here we pretend the biodiversity indicator is expensive, count
how often it is called, and compare:

  A) Baseline: NSGA-II with the true indicator for every evaluation.
  B) Surrogate-assisted: a regression model learns the indicator from a
     few hundred true evaluations. NSGA-II then runs on the surrogate.
     After each round, the final front is checked with the true indicator
     and added to the training data (simple adaptive loop).

Revenue is cheap and always computed exactly.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from landscape import make_landscape, NATURAL, PASTURE
from objectives import agricultural_revenue, biodiversity_score, evaluate
from nsga2 import nsga2, random_map
from metrics import hypervolume_2d, non_dominated

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
REF = (0.0, 0.0)  # hypervolume reference point (worse than any solution)


class CountingIndicator:
    """Wraps the 'expensive' biodiversity indicator and counts its calls."""

    def __init__(self, landscape):
        self.landscape, self.calls = landscape, 0

    def __call__(self, lu):
        self.calls += 1
        return biodiversity_score(lu, self.landscape["habitat_map"])


def features(lu):
    """Input of the surrogate: one-hot map (is natural?, is pasture?) per cell."""
    return np.concatenate([(lu == NATURAL).ravel(), (lu == PASTURE).ravel()]).astype(float)


def run_baseline(ls, generations, seed):
    true_bio = CountingIndicator(ls)
    ev = lambda lu, l: (agricultural_revenue(lu, l["yield_map"]), true_bio(lu))
    t = time.time()
    _, F = nsga2(ls, ev, generations=generations, seed=seed, verbose=False)
    return F, true_bio.calls, time.time() - t


def run_surrogate(ls, generations, seed, n_initial=300, rounds=3):
    rng = np.random.default_rng(seed)
    true_bio = CountingIndicator(ls)
    t = time.time()

    # 1. Initial training data: random maps evaluated with the true indicator
    maps = [random_map(ls, rng) for _ in range(n_initial)]
    X = [features(m) for m in maps]
    y = [true_bio(m) for m in maps]

    seeds, archive_F = None, []
    for r in range(rounds):
        # 2. Train the surrogate (ridge regression)
        model = Ridge(alpha=1.0).fit(np.array(X), np.array(y))
        ev = lambda lu, l: (agricultural_revenue(lu, l["yield_map"]),
                            float(model.predict(features(lu)[None])[0]))
        # 3. Optimize on the surrogate (warm-started from the previous round)
        front, _ = nsga2(ls, ev, generations=generations // rounds,
                         seed=seed + r, verbose=False, initial_pop=seeds)
        # 4. Check the front with the true indicator and add it to the data
        for m in front:
            X.append(features(m))
            y.append(true_bio(m))
            archive_F.append(evaluate(m, ls))
        seeds = front
    F = np.array(archive_F)
    F = F[non_dominated(F)]
    return F, true_bio.calls, time.time() - t


if __name__ == "__main__":
    ls = make_landscape(n=30, seed=42)
    gens = 200
    F_base, calls_base, t_base = run_baseline(ls, gens, seed=0)
    F_sur, calls_sur, t_sur = run_surrogate(ls, gens, seed=0)
    hv_base, hv_sur = hypervolume_2d(F_base, REF), hypervolume_2d(F_sur, REF)

    print(f"Baseline : {calls_base:6d} true evaluations, hypervolume {hv_base:,.0f}, {t_base:.0f}s")
    print(f"Surrogate: {calls_sur:6d} true evaluations, hypervolume {hv_sur:,.0f} "
          f"({100 * hv_sur / hv_base:.1f}% of baseline), {t_sur:.0f}s")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for F, label, c in [(F_base, f"True indicator ({calls_base:,} true evaluations)", "#3366AA"),
                        (F_sur, f"Surrogate-assisted ({calls_sur:,} true evaluations)", "#C0392B")]:
        o = np.argsort(F[:, 0])
        ax.plot(F[o, 0], F[o, 1], "o-", ms=3, lw=1, color=c, label=label)
    ax.set_xlabel("Agricultural revenue (relative units)")
    ax.set_ylabel("Biodiversity score (true indicator)")
    ax.set_title("Surrogate-assisted optimization")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "5_surrogate.png"), dpi=150)
