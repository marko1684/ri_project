"""
Exhaustive search (true brute force).

Enumerates every one of the 2^m subsets of the m internal grid edges, 
Returns the shortest valid partition.
It is only feasible for very small instances (roughly up to ~20 internal edges).
It is guaranteed to find the optimal solution.
"""

import time
from typing import Optional, Tuple

from hanan_grid import ProblemInstance, HananGrid
from partition import Solution, build_solution, total_length, is_valid_partition


def solve_exhaustive(instance: ProblemInstance, grid: HananGrid,
                     time_limit_seconds: Optional[float] = None) -> Tuple[Solution, float]:
    """Try all edge subsets and return best valid solution."""
    num_edges = len(grid.internal_edges)
    start = time.time()

    best_selection = None
    best_length = float("inf")

    for mask in range(2 ** num_edges):
        if time_limit_seconds is not None and time.time() - start > time_limit_seconds:
            break

        selection = [(mask >> i) & 1 for i in range(num_edges)]
        length = total_length(selection, grid)
        if length < best_length and is_valid_partition(selection, grid):
            best_length = length
            best_selection = selection

    if best_selection is None:
        best_selection = [1] * num_edges

    return build_solution(best_selection, grid), time.time() - start
