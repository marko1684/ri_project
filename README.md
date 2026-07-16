# ri_project

University project for the **Computational Intelligence** course.

Repository for solving the **Minimum Rectangle Partition with Interior Points**
problem: given an axis-aligned rectangle and a set of points lying strictly
inside it, find the shortest set of axis-parallel cut segments that partitions
the rectangle into smaller rectangles such that **every interior point lies on
at least one cut** (i.e. every point is "covered" by the partition).

The project implements and compares several approaches — exact solvers
(brute force, branch and bound), a greedy construction heuristic, and two
metaheuristics (simulated annealing and a genetic algorithm) — together with
instance generation, plotting, and an experimentation runner.

## The problem

**Input.** A rectangle `R = [x_min, x_max] × [y_min, y_max]` and a set of
interior points `P = {p_1, …, p_n}` with `p_i ∈ int(R)`.

**Output.** A set of axis-parallel segments inside `R` of minimum total length
such that:

- the segments partition `R` into rectangles,
- every point `p_i` lies on at least one segment.

**Why the Hanan grid.** A classical result (Hanan, 1966) shows that for this
class of rectilinear partitioning problems there always exists an optimal
solution whose segments lie on the *Hanan grid* — the grid obtained by drawing
a horizontal and a vertical line through every interior point and clipping to
the rectangle. The search therefore reduces to selecting a subset of the grid's
internal edges, which is what every solver in this repository does.

A valid selection must satisfy the following per-vertex constraints
(see `partition.py`):

1. every interior point has internal degree `≥ 2` (it lies on a cut),
2. no vertex has degree `1` (no dangling segment),
3. no vertex outside the rectangle's corners is an L-junction (exactly one
   horizontal and one vertical edge), which would create a reflex corner,
4. all selected edges connect back to the rectangle boundary.

## Repository structure

| File                       | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `hanan_grid.py`            | Geometric model: `Point`, `Rectangle`, `ProblemInstance`, and `HananGrid` construction. |
| `instances.py`             | Test instance generators: random, clustered, and coordinate-sharing.       |
| `partition.py`             | Scoring (`total_length`), validity checks, violation counting, penalized cost, and the `Solution` dataclass. |
| `exhaustive_search.py`     | True brute force: enumerates all `2^m` edge subsets. Optimal but feasible only for tiny instances. |
| `branch_and_bound.py`      | Exact search with cost and feasibility pruning; reaches larger instances than brute force. |
| `greedy.py`                | Greedy construction: start from the full grid and drop the longest removable edges. |
| `simulated_annealing.py`   | SA over the edge selection with single-bit, multi-bit, and line-flip moves; supports restarts. |
| `genetic_algorithm.py`     | GA with binary chromosomes, tournament selection, uniform/two-point crossover, and line-aware mutation. |
| `plotting.py`              | Matplotlib plotting of instances, partitions, comparisons, and convergence curves. |
| `experimentation.py`       | Experiment runner: brute force, method comparison, and convergence plots. Writes figures to `results/`. |
| `results/`                 | Generated figures (brute-force optima, comparisons, convergence).          |
| `seminarski.pdf`           | Project paper (Serbian).                                                    |

## Solvers

### Exact

- **Exhaustive search** (`solve_exhaustive`) — enumerates every subset of the
  `m` internal edges and returns the shortest valid one. Guaranteed optimal but
  only feasible up to roughly `m ≈ 20` edges.
- **Branch and bound** (`solve_branch_and_bound`) — same search space, pruned by
  (a) a cost bound initialized from the greedy solution and (b) a feasibility
  bound that abandons a branch as soon as a vertex can no longer satisfy the
  partition constraints. Reaches noticeably larger instances than brute force.

### Heuristic

- **Greedy construction** (`greedy_construction`) — starts from the full grid
  (always a valid partition) and removes edges from longest to shortest,
  keeping each removal only while the partition stays valid. Cheap and always
  valid; used as the initial bound for branch and bound and to seed the GA.

### Metaheuristics

Both metaheuristics minimize the *penalized cost*
`total_length + penalty_weight * count_violations`, which lets the search move
through invalid selections while being pushed back toward feasibility. Only
valid selections are recorded as the returned best.

- **Simulated annealing** (`solve_simulated_annealing`,
  `solve_sa_with_restarts`) — starts from a random selection, accepts worse
  neighbors with probability `exp(-Δ / T)`, and cools geometrically. Neighbors
  are drawn from three move types: flip a single edge, flip a few edges, or
  flip a whole grid line.
- **Genetic algorithm** (`solve_genetic_algorithm`) — a chromosome is the binary
  edge vector; fitness is the negated penalized cost. The initial population
  mixes greedy individuals (valid anchors) with random ones (diversity). Each
  generation keeps the elite and fills the rest via tournament selection,
  uniform or two-point crossover, and bit-flip / line-aware mutation.

## Usage

### Requirements

- Python 3.10+
- `matplotlib` (for plotting)

### Quick start

Build an instance, construct its Hanan grid, and run any solver:

```python
from hanan_grid import build_hanan_grid
from instances import random_instance
from genetic_algorithm import solve_genetic_algorithm

instance = random_instance(num_points=12, seed=42)
grid = build_hanan_grid(instance)

solution, elapsed, stats = solve_genetic_algorithm(instance, grid, seed=123)
print(f"length={solution.total_length:.2f}  valid={solution.is_valid}  time={elapsed:.2f}s")
```

### Run all experiments

```bash
python experimentation.py
```

This runs brute force on the smallest instances, branch and bound on slightly
larger ones, compares every method on instances of 2, 3, 8, and 15 points, and
generates SA/GA convergence curves. All figures are written to `results/`.

### Generate a custom instance

```python
from instances import random_instance, clustered_instance, shared_coordinates_instance

# Fully random points with distinct x and y coordinates (hardest case).
inst = random_instance(num_points=10, seed=1)

# Points grouped around a few cluster centers (some share coordinates).
inst = clustered_instance(num_points=10, num_clusters=3, seed=1)

# Points drawn from a limited coordinate pool (many share a row or column).
inst = shared_coordinates_instance(num_points=10, sharing_ratio=0.3, seed=1)
```

## Notes

- Instances with all-distinct point coordinates are the hardest, since no cut
  can be shared between two points. The generators in `instances.py` cover a
  range of structures from fully random to heavily coordinate-sharing.
- The metaheuristics are stochastic; pass a `seed` for reproducibility.
- All solvers return a `Solution` (edge selection, total length, validity) and
  the elapsed time; the metaheuristics additionally return a `stats` dict with
  convergence histories used by `plotting.plot_convergence`.

## Presentation

[Link](https://canva.link/yglurnbh04kxf4y)
