# Land Use Pareto Demo

A small learning prototype of **multi-objective spatial optimization of land use**. It uses a self-written NSGA-II on a synthetic raster landscape.

The goal was to understand, hands-on, how Pareto frontiers show trade-offs between agricultural production and biodiversity conservation. This is not a calibrated model. All data are synthetic and the indicators are deliberately simple.

*This prototype was developed with the help of an AI coding assistant, as a learning exercise.*

## Problem

- **Landscape:** a 30 × 30 raster. Each cell is assigned one of three land uses: cropland, pasture or natural vegetation.
- **Inputs:** spatially autocorrelated synthetic maps of crop yield potential and habitat suitability, plus a protected area that must stay natural (constraint).
- **Objective 1 – Agricultural revenue (maximize):** yield × price, summed over cropland and pasture cells.
- **Objective 2 – Biodiversity score (maximize):** habitat suitability of natural cells, with a bonus for natural neighbours. This rewards compact patches and penalizes fragmentation. Pasture keeps a small share of habitat value.

![Inputs](figures/1_inputs.png)

## Method

NSGA-II (Deb et al., 2002) implemented from scratch in `src/nsga2.py`:

- fast non-dominated sorting and crowding distance;
- binary tournament selection and elitist survival;
- **spatial operators:**
  - **block crossover:** swaps a rectangular area between two parent maps;
  - **neighbour-based mutation:** a cell copies the land use of a neighbouring cell. Both keep solutions spatially coherent;
- a repair step that enforces the protected-area constraint.

## Results

The Pareto front shows the trade-off clearly. Gaining agricultural revenue costs increasingly more biodiversity towards the right end of the front.

![Pareto front](figures/2_pareto_front.png)

Three solutions from the front: maximum revenue, the knee point (closest to the ideal point after normalization) and maximum biodiversity.

![Example maps](figures/3_example_maps.png)

### Simple uncertainty test

The yield map was perturbed with ±15% random noise (5 samples), and the optimization was re-run for each sample. The fronts stay similar in the balanced middle part, and they diverge more at the high-revenue end, which depends most on the yield values.

Some of the spread also comes from the stochastic optimizer itself (different random seeds and a limited number of generations). A proper analysis would need more samples and repeated runs, as in Hildemann & Verstegen (2021).

![Uncertainty](figures/4_uncertainty.png)

## Limitations and possible next steps

- Synthetic data and simplified indicators. Real indicators would come from, for example, species distribution models and yield or price data.
- Each objective evaluation is cheap here. With expensive indicator models, **learning-based surrogates** could reduce computation time.
- A full Monte Carlo over input uncertainty is costly. More efficient approaches exist (e.g. exterior sampling with seeding).
- Comparison with a library implementation (e.g. pymoo) and with other algorithms.

## Run

```bash
pip install -r requirements.txt
python src/run.py                      # full run (~4 min)
python src/run.py --skip-uncertainty   # without the uncertainty test (~1 min)
```

## References

- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.
- Hildemann, M., & Verstegen, J. A. (2021). Quantifying uncertainty in Pareto fronts arising from spatial data. *Environmental Modelling & Software*.
