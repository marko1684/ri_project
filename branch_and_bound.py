"""
Branch and bound exact solver.

Same search space as exhaustive_search, but two kinds of pruning let it reach
noticeably larger instances:

  - cost bound: abandon a branch once its partial length reaches the best valid
    length found so far (the greedy construction gives the initial bound),
  - feasibility bound: abandon a branch once a vertex can no longer satisfy the
    partition constraints (a point that can never reach degree 2, an unavoidable
    dangling end, or a decided L-junction).
"""

import time
from typing import Optional, Tuple, List

from hanan_grid import ProblemInstance, HananGrid
from partition import (
    Solution, build_solution, total_length, is_valid_partition,
    orientation_degrees, is_rectangle_corner,
)
from greedy import greedy_construction


def _can_still_satisfy(partial_selection: List[int], grid: HananGrid) -> bool:
    """Return False if some vertex is already doomed given the decided edges."""
    point_vertices = set(grid.point_vertex_indices)
    decided = len(partial_selection)

    for vertex_idx in range(len(grid.vertices)):
        edges = grid.vertex_to_internal_edges.get(vertex_idx, [])
        selected = sum(partial_selection[e] for e in edges if e < decided)
        undecided = sum(1 for e in edges if e >= decided)

        if vertex_idx in point_vertices and selected + undecided < 2:
            return False

        if undecided == 0:
            if not grid.is_boundary_vertex[vertex_idx] and selected == 1:
                return False
            padded = partial_selection + [0] * (len(grid.internal_edges) - decided)
            horizontal, vertical = orientation_degrees(vertex_idx, padded, grid)
            if horizontal == 1 and vertical == 1 and not is_rectangle_corner(vertex_idx, grid):
                return False

    return True


def solve_branch_and_bound(instance: ProblemInstance, grid: HananGrid,
                           time_limit_seconds: Optional[float] = None
                           ) -> Tuple[Solution, float]:
    """Exact search with cost and feasibility pruning. Returns (solution, seconds)."""
    num_edges = len(grid.internal_edges)
    start = time.time()

    greedy = greedy_construction(grid)
    best = {"selection": greedy[:], "length": total_length(greedy, grid)}

    def recurse(edge_idx: int, selection: List[int], length: float):
        if time_limit_seconds is not None and time.time() - start > time_limit_seconds:
            return
        if length >= best["length"] or not _can_still_satisfy(selection, grid):
            return

        if edge_idx == num_edges:
            if length < best["length"] and is_valid_partition(selection, grid):
                best["length"] = length
                best["selection"] = selection[:]
            return

        selection.append(0)                       # cheaper branch first
        recurse(edge_idx + 1, selection, length)
        selection.pop()

        extended = length + grid.edge_lengths[edge_idx]
        if extended < best["length"]:
            selection.append(1)
            recurse(edge_idx + 1, selection, extended)
            selection.pop()

    recurse(0, [], 0.0)
    return build_solution(best["selection"], grid), time.time() - start
