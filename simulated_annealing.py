"""
Simulated annealing over the Hanan-grid edge selection.

The search starts from a random selection and accepts worse neighbors with the
usual probability exp(-delta / temperature), which cools geometrically. Invalid
selections are allowed via the penalized cost, but only valid ones are recorded
as the returned best. Neighbors are drawn from three move types with fixed
probabilities so the search mixes fine and coarse steps.
"""

import random
import math
import time
from typing import List, Optional, Tuple

from hanan_grid import ProblemInstance, HananGrid
from partition import (
    Solution, build_solution, total_length, is_valid_partition, penalized_cost,
)


def random_selection(num_edges: int) -> List[int]:
    """Random binary edge selection."""
    return [random.randint(0, 1) for _ in range(num_edges)]


def random_neighbor(selection: List[int], grid: HananGrid) -> List[int]:
    """Return a neighbor by one of three moves: flip a single edge, flip a few
    edges, or flip a whole grid line (all edges along one x or y)."""
    neighbor = selection[:]
    num_edges = len(neighbor)
    roll = random.random()

    if roll < 0.5:
        idx = random.randrange(num_edges)
        neighbor[idx] ^= 1
    elif roll < 0.75:
        for idx in random.sample(range(num_edges), random.randint(2, min(3, num_edges))):
            neighbor[idx] ^= 1
    else:
        _flip_grid_line(neighbor, grid)

    return neighbor


def _flip_grid_line(selection: List[int], grid: HananGrid):
    """Flip every internal edge lying on one randomly chosen interior grid line."""
    horizontal = random.random() < 0.5
    coords = grid.y_coords[1:-1] if horizontal else grid.x_coords[1:-1]
    if not coords:
        return
    target = random.choice(coords)
    for i, (v1_idx, v2_idx) in enumerate(grid.internal_edges):
        v1, v2 = grid.vertices[v1_idx], grid.vertices[v2_idx]
        on_line = (v1.y == target and v2.y == target) if horizontal else \
                  (v1.x == target and v2.x == target)
        if on_line:
            selection[i] ^= 1


def solve_simulated_annealing(instance: ProblemInstance, grid: HananGrid,
                              initial_temperature: float = 500.0,
                              cooling_rate: float = 0.9995,
                              min_temperature: float = 0.01,
                              max_iterations: int = 20000,
                              seed: Optional[int] = None) -> Tuple[Solution, float, dict]:
    """Run simulated annealing from a random starting potential solution. Returns (best valid solution,
    elapsed seconds, stats)."""
    if seed is not None:
        random.seed(seed)

    start = time.time()
    current = random_selection(len(grid.internal_edges))
    current_cost = penalized_cost(current, grid)

    best_valid = None
    best_valid_length = float("inf")
    cost_history = []

    temperature = initial_temperature
    iteration = 0
    while temperature > min_temperature and iteration < max_iterations:

        neighbor = random_neighbor(current, grid)
        neighbor_cost = penalized_cost(neighbor, grid)
        delta = neighbor_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current, current_cost = neighbor, neighbor_cost

        if is_valid_partition(current, grid):
            length = total_length(current, grid)
            if length < best_valid_length:
                best_valid_length, best_valid = length, current[:]
        
        
        if iteration % 500 == 0:
            cost_history.append(best_valid_length if best_valid else current_cost)

        temperature *= cooling_rate
        iteration += 1

    selection = best_valid if best_valid is not None else current
    stats = {"iterations": iteration, "cost_history": cost_history}
    return build_solution(selection, grid), time.time() - start, stats


def solve_sa_with_restarts(instance: ProblemInstance, grid: HananGrid,
                           num_restarts: int = 5,
                           max_iterations: int = 15000,
                           seed: Optional[int] = None) -> Tuple[Solution, float, dict]:
    """Run simulated annealing several times with increasing initial temperature
    and keep the best valid solution."""
    if seed is not None:
        random.seed(seed)

    start = time.time()
    best_solution = None
    best_length = float("inf")
    all_stats = []

    for restart in range(num_restarts):
        solution, _, stats = solve_simulated_annealing(
            instance, grid,
            initial_temperature=500.0 * (1 + restart * 0.5),
            cooling_rate=0.9993,
            max_iterations=max_iterations,
            seed=(seed + restart * 7) if seed is not None else None,
        )
        all_stats.append(stats)
        if solution.is_valid and solution.total_length < best_length:
            best_length, best_solution = solution.total_length, solution

    if best_solution is None:
        best_solution = build_solution(random_selection(len(grid.internal_edges)), grid)

    stats = {"num_restarts": len(all_stats), "restart_stats": all_stats}
    if all_stats:
        stats["cost_history"] = all_stats[0].get("cost_history", [])
    return best_solution, time.time() - start, stats
