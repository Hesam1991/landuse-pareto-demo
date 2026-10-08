"""Experiment 2: reusing solutions across scenarios (seeding).

The project needs Pareto fronts for several climate and socio-economic
scenarios. Re-running the optimization from scratch for every scenario is
expensive. The simplest way to reuse what was learned in one scenario is to
start the next optimization from its Pareto front (seeding).

Scenario A : current conditions.
Scenario B : a "drier climate" in which crop yield drops by up to 40% from
             south to north (a simple spatial gradient).

We compare, for scenario B, NSGA-II started from random maps versus started
from the scenario-A front, by tracking the hypervolume per generation.
This is plain seeding, not machine learning; learning-based approaches
(e.g. a model conditioned on the scenario) are a possible next step.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from landscape import make_landscape
from objectives import evaluate
from nsga2 import nsga2
from metrics import hypervolume_2d

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
REF = (0.0, 0.0)


def drier_scenario(ls, max_loss=0.4):
    """Yield decreases linearly from 0% (south) to max_loss (north)."""
    n = ls["n"]
    factor = 1 - max_loss * np.linspace(1, 0, n)[:, None]  # row 0 = north
    scen = dict(ls)
    scen["yield_map"] = ls["yield_map"] * factor
    return scen


def run_with_history(ls, generations, seed, initial_pop=None):
    history = []
    cb = lambda g, pop, F: history.append(hypervolume_2d(F, REF))
    front, F = nsga2(ls, evaluate, generations=generations, seed=seed,
                     verbose=False, initial_pop=initial_pop, callback=cb)
    return front, np.array(history)


if __name__ == "__main__":
    ls_a = make_landscape(n=30, seed=42)
    ls_b = drier_scenario(ls_a)
    gens, seeds = 100, [0, 1, 2]

    front_a, _ = nsga2(ls_a, evaluate, generations=200, seed=0, verbose=False)
    print("scenario A optimized")

    scratch, seeded = [], []
    for s in seeds:
        scratch.append(run_with_history(ls_b, gens, seed=10 + s)[1])
        seeded.append(run_with_history(ls_b, gens, seed=10 + s, initial_pop=front_a)[1])
        print(f"scenario B, repetition {s + 1}/{len(seeds)} done")
    scratch, seeded = np.array(scratch), np.array(seeded)

    final = scratch.mean(0)[-1]
    reach = lambda h: int(np.argmax(h >= 0.99 * final)) + 1 if np.any(h >= 0.99 * final) else None
    print(f"From scratch: 99% of its final hypervolume after {reach(scratch.mean(0))} generations")
    print(f"Seeded      : the same level after {reach(seeded.mean(0))} generations")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    g = np.arange(1, gens + 1)
    for H, label, c in [(scratch, "From scratch (random start)", "#3366AA"),
                        (seeded, "Seeded with scenario-A front", "#C0392B")]:
        ax.plot(g, H.mean(0), color=c, label=label)
        ax.fill_between(g, H.min(0), H.max(0), color=c, alpha=0.15)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume (scenario B)")
    ax.set_title("Reusing solutions across scenarios (mean and range of 3 runs)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "6_scenario_seeding.png"), dpi=150)
