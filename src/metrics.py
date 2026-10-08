"""Quality metrics for Pareto fronts."""
import numpy as np


def hypervolume_2d(F, ref):
    """Hypervolume of a 2-objective front (both objectives maximized).

    It is the area dominated by the front and bounded by the reference
    point `ref` (a point worse than every solution). A larger hypervolume
    means a front that is both closer to the ideal and better spread.
    """
    F = np.asarray(F, dtype=float)
    F = F[np.all(F > ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(-F[:, 0])]          # sort by objective 1, descending
    hv, best_y = 0.0, ref[1]
    for x, y in F:
        if y > best_y:                   # only non-dominated steps add area
            hv += (x - ref[0]) * (y - best_y)
            best_y = y
    return hv


def non_dominated(F):
    """Indices of the non-dominated rows of F (maximization)."""
    F = np.asarray(F)
    keep = []
    for i, f in enumerate(F):
        dominated = np.any(np.all(F >= f, axis=1) & np.any(F > f, axis=1))
        if not dominated:
            keep.append(i)
    return np.array(keep)
