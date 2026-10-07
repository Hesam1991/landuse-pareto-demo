"""Objective functions (impact indicators).

Two conflicting objectives, loosely following indicators used in land use
trade-off studies (e.g. agricultural revenue and species richness / habitat):

1. Agricultural revenue  -> maximize
2. Biodiversity score    -> maximize

Both are simplified on purpose: this is a learning prototype, not a
calibrated model.
"""
import numpy as np
from landscape import CROP, PASTURE, NATURAL

PRICE_CROP = 1.0      # relative revenue per unit yield on cropland
PRICE_PASTURE = 0.35  # pasture earns less per unit yield than crops
NEIGHBOR_BONUS = 1.0  # extra biodiversity value for natural cells in compact patches


def agricultural_revenue(lu, yield_map):
    """Sum of revenue over all cropland and pasture cells."""
    revenue = np.where(lu == CROP, PRICE_CROP * yield_map, 0.0)
    revenue += np.where(lu == PASTURE, PRICE_PASTURE * yield_map, 0.0)
    return revenue.sum()


def natural_neighbor_share(lu):
    """Share of the 4 direct neighbours of each cell that are natural."""
    nat = (lu == NATURAL).astype(float)
    padded = np.pad(nat, 1, mode="constant")
    neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                 padded[1:-1, :-2] + padded[1:-1, 2:])
    return neighbors / 4.0


def biodiversity_score(lu, habitat_map):
    """Habitat quality of natural cells, rewarding compact (less fragmented)
    natural patches. Pasture keeps a small part of the habitat value."""
    nat = lu == NATURAL
    share = natural_neighbor_share(lu)
    score = np.where(nat, habitat_map * (1.0 + NEIGHBOR_BONUS * share), 0.0)
    score += np.where(lu == PASTURE, 0.2 * habitat_map, 0.0)
    return score.sum()


def evaluate(lu, landscape):
    """Return both objectives for one land use map."""
    return (agricultural_revenue(lu, landscape["yield_map"]),
            biodiversity_score(lu, landscape["habitat_map"]))
