"""Synthetic landscape generation.

Creates a small raster landscape with spatially autocorrelated input maps,
so the optimization problem has realistic spatial structure (clusters of
fertile soil, clusters of valuable habitat) instead of pure noise.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# Land use classes
CROP, PASTURE, NATURAL = 0, 1, 2
CLASS_NAMES = {CROP: "Cropland", PASTURE: "Pasture", NATURAL: "Natural vegetation"}


def smooth_field(shape, sigma, rng):
    """Spatially autocorrelated random field scaled to [0, 1]."""
    field = gaussian_filter(rng.normal(size=shape), sigma=sigma)
    return (field - field.min()) / (field.max() - field.min())


def make_landscape(n=30, seed=42):
    """Return a dict with the input maps of the synthetic landscape.

    yield_map      : relative crop yield potential per cell (0-1)
    habitat_map    : habitat suitability for species per cell (0-1)
    protected_mask : cells that must stay natural (e.g. a protected area)
    """
    rng = np.random.default_rng(seed)
    yield_map = smooth_field((n, n), sigma=4, rng=rng)
    # Habitat suitability is partly anti-correlated with yield: the most
    # fertile land is often also valuable habitat, but not everywhere.
    habitat_raw = 0.5 * smooth_field((n, n), sigma=3, rng=rng) + 0.5 * (1 - yield_map)
    habitat_map = (habitat_raw - habitat_raw.min()) / (habitat_raw.max() - habitat_raw.min())

    # A circular protected area in the cell with the highest habitat value
    cy, cx = np.unravel_index(np.argmax(gaussian_filter(habitat_map, 2)), (n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    protected_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= (n // 8) ** 2

    return {"yield_map": yield_map, "habitat_map": habitat_map,
            "protected_mask": protected_mask, "n": n}
