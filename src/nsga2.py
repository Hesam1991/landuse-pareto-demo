"""A compact NSGA-II implementation for raster land use allocation.

NSGA-II (Deb et al., 2002) in four ideas:
1. Non-dominated sorting: rank solutions into Pareto fronts.
2. Crowding distance: within a front, prefer solutions in less crowded
   regions, so the front stays well spread.
3. Binary tournament selection on (rank, crowding distance).
4. Elitism: parents and offspring compete together for survival.

The genetic operators are adapted to maps: crossover swaps a rectangular
block between two parent maps, and mutation copies the land use of a
neighbouring cell, which keeps solutions spatially coherent.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from landscape import CROP, PASTURE, NATURAL


# ---------------------------------------------------------------- sorting
def dominates(a, b):
    """True if a dominates b (all objectives maximized)."""
    return np.all(a >= b) and np.any(a > b)


def non_dominated_sort(F):
    """Return a list of fronts; each front is a list of indices into F."""
    n = len(F)
    dominated_by = [[] for _ in range(n)]   # solutions that i dominates
    domination_count = np.zeros(n, dtype=int)  # how many dominate i
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(F[i], F[j]):
                dominated_by[i].append(j)
            elif dominates(F[j], F[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(F):
    """Crowding distance of each solution within one front."""
    n, m = F.shape
    dist = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        order = np.argsort(F[:, k])
        dist[order[0]] = dist[order[-1]] = np.inf
        span = F[order[-1], k] - F[order[0], k]
        if span == 0:
            continue
        dist[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return dist


# --------------------------------------------------------------- operators
def random_map(landscape, rng):
    """Patchy random land use map (thresholded smooth noise)."""
    n = landscape["n"]
    field = gaussian_filter(rng.normal(size=(n, n)), sigma=rng.uniform(1, 3))
    q1, q2 = np.quantile(field, sorted(rng.uniform(0.2, 0.8, 2)))
    lu = np.where(field < q1, CROP, np.where(field < q2, PASTURE, NATURAL))
    return repair(lu, landscape)


def repair(lu, landscape):
    """Enforce the constraint: protected cells stay natural."""
    lu = lu.copy()
    lu[landscape["protected_mask"]] = NATURAL
    return lu


def block_crossover(p1, p2, rng):
    """Swap a random rectangle between two parent maps."""
    n = p1.shape[0]
    h, w = rng.integers(n // 5, n // 2, size=2)
    y, x = rng.integers(0, n - h), rng.integers(0, n - w)
    c1, c2 = p1.copy(), p2.copy()
    c1[y:y + h, x:x + w] = p2[y:y + h, x:x + w]
    c2[y:y + h, x:x + w] = p1[y:y + h, x:x + w]
    return c1, c2


def spatial_mutation(lu, rng, rate=0.01, p_random=0.1):
    """Each mutated cell copies a random neighbour's class (keeps patches
    coherent); with small probability it gets a random class instead."""
    n = lu.shape[0]
    child = lu.copy()
    cells = np.argwhere(rng.random(lu.shape) < rate)
    moves = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
    for y, x in cells:
        if rng.random() < p_random:
            child[y, x] = rng.integers(0, 3)
        else:
            dy, dx = moves[rng.integers(4)]
            child[y, x] = lu[np.clip(y + dy, 0, n - 1), np.clip(x + dx, 0, n - 1)]
    return child


# ------------------------------------------------------------------- main
def survive(pop, F, size):
    """Elitist survival: fill the next population front by front, breaking
    ties in the last front with crowding distance."""
    chosen, rank, crowd = [], np.zeros(len(F), int), np.zeros(len(F))
    for r, front in enumerate(non_dominated_sort(F)):
        cd = crowding_distance(F[front])
        rank[front], crowd[front] = r, cd
        if len(chosen) + len(front) <= size:
            chosen += front
        else:
            order = np.argsort(-cd)
            chosen += [front[i] for i in order[:size - len(chosen)]]
            break
    idx = np.array(chosen)
    return [pop[i] for i in idx], F[idx], rank[idx], crowd[idx]


def tournament(rank, crowd, rng):
    """Binary tournament: lower rank wins, then larger crowding distance."""
    a, b = rng.integers(0, len(rank), 2)
    if rank[a] != rank[b]:
        return a if rank[a] < rank[b] else b
    return a if crowd[a] > crowd[b] else b


def nsga2(landscape, evaluate, pop_size=100, generations=200,
          p_crossover=0.9, mutation_rate=0.01, seed=0, verbose=True):
    """Run NSGA-II and return the final population and objective values."""
    rng = np.random.default_rng(seed)
    pop = [random_map(landscape, rng) for _ in range(pop_size)]
    F = np.array([evaluate(lu, landscape) for lu in pop])
    pop, F, rank, crowd = survive(pop, F, pop_size)

    for gen in range(generations):
        offspring = []
        while len(offspring) < pop_size:
            p1 = pop[tournament(rank, crowd, rng)]
            p2 = pop[tournament(rank, crowd, rng)]
            if rng.random() < p_crossover:
                c1, c2 = block_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            offspring += [repair(spatial_mutation(c, rng, mutation_rate), landscape)
                          for c in (c1, c2)]
        F_off = np.array([evaluate(lu, landscape) for lu in offspring])
        pop, F, rank, crowd = survive(pop + offspring, np.vstack([F, F_off]), pop_size)
        if verbose and (gen + 1) % 50 == 0:
            print(f"generation {gen + 1}: {np.sum(rank == 0)} solutions on the first front")
    first = rank == 0
    return [p for p, f in zip(pop, first) if f], F[first]
