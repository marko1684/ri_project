"""
Scoring and validity of a partition on the Hanan grid.

A partition is represented as a binary vector over the grid's internal edges
(1 = selected, 0 = not). This module measures its total length and checks the
constraints that make it a valid rectangular partition:

  1. every interior point has internal degree >= 2 (it lies on a cut),
  2. no vertex has degree 1 (no dangling segment),
  3. no vertex outside the rectangle's corners is an L-junction (exactly one
     horizontal and one vertical edge), which would create a reflex corner,
  4. all selected edges connect back to the rectangle boundary.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from collections import deque

from hanan_grid import HananGrid


@dataclass
class Solution:
    edge_selection: List[int] = field(default_factory=list)
    total_length: float = 0.0
    is_valid: bool = False

    def copy(self) -> "Solution":
        return Solution(list(self.edge_selection), self.total_length, self.is_valid)


def total_length(edge_selection: List[int], grid: HananGrid) -> float:
    """Sum the lengths of the selected internal edges."""
    return sum(grid.edge_lengths[i] for i, sel in enumerate(edge_selection) if sel)


def orientation_degrees(vertex_idx: int, edge_selection: List[int],
                        grid: HananGrid) -> Tuple[int, int]:
    """Return (horizontal_degree, vertical_degree) at a vertex, counting the
    always-present boundary edges plus the selected internal edges."""
    vertex = grid.vertices[vertex_idx]
    horizontal = vertical = 0

    def tally(edge, other_idx):
        nonlocal horizontal, vertical
        other = grid.vertices[other_idx]
        if other.y == vertex.y:
            horizontal += 1
        else:
            vertical += 1

    for edge_idx in grid.vertex_to_boundary_edges.get(vertex_idx, []):
        v1, v2 = grid.boundary_edges[edge_idx]
        tally(edge_idx, v2 if v1 == vertex_idx else v1)

    for edge_idx in grid.vertex_to_internal_edges.get(vertex_idx, []):
        if edge_selection[edge_idx]:
            v1, v2 = grid.internal_edges[edge_idx]
            tally(edge_idx, v2 if v1 == vertex_idx else v1)

    return horizontal, vertical


def is_rectangle_corner(vertex_idx: int, grid: HananGrid) -> bool:
    """Check whether a vertex is one of the four corners of the rectangle."""
    v = grid.vertices[vertex_idx]
    return (v.x in (grid.x_coords[0], grid.x_coords[-1]) and
            v.y in (grid.y_coords[0], grid.y_coords[-1]))


def count_violations(edge_selection: List[int], grid: HananGrid) -> int:
    """Count local constraint violations. Zero means the selection satisfies the
    per-vertex rules; it is used both for validity and as a penalty for the
    metaheuristics."""
    violations = 0
    point_vertices = set(grid.point_vertex_indices)

    for vertex_idx in range(len(grid.vertices)):
        horizontal, vertical = orientation_degrees(vertex_idx, edge_selection, grid)

        if vertex_idx in point_vertices:
            internal_degree = sum(
                edge_selection[e] for e in grid.vertex_to_internal_edges.get(vertex_idx, [])
            )
            if internal_degree < 2:
                violations += 2 - internal_degree

        if horizontal + vertical == 1:
            violations += 1
        elif horizontal == 1 and vertical == 1 and not is_rectangle_corner(vertex_idx, grid):
            violations += 1

    return violations


def reaches_boundary(edge_selection: List[int], grid: HananGrid) -> bool:
    """Check that every vertex touched by a selected edge is connected, through
    selected edges, to the rectangle boundary (BFS from the boundary)."""
    adj = {}
    involved = set()
    for edge_idx, sel in enumerate(edge_selection):
        if not sel:
            continue
        v1, v2 = grid.internal_edges[edge_idx]
        adj.setdefault(v1, set()).add(v2)
        adj.setdefault(v2, set()).add(v1)
        involved.update((v1, v2))

    if not involved:
        return True

    starts = [v for v in involved if grid.is_boundary_vertex[v]]
    if not starts:
        return False

    visited = set(starts)
    queue = deque(starts)
    while queue:
        current = queue.popleft()
        for neighbor in adj.get(current, ()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return involved.issubset(visited)


def penalized_cost(edge_selection: List[int], grid: HananGrid,
                   penalty_weight: float = 1000.0) -> float:
    """Objective the metaheuristics minimize: total length plus a penalty per
    constraint violation. This lets a search move through invalid selections
    while being pushed back toward feasibility."""
    return total_length(edge_selection, grid) + count_violations(edge_selection, grid) * penalty_weight


def is_valid_partition(edge_selection: List[int], grid: HananGrid) -> bool:
    """A selection is valid when it has no local violations and stays connected
    to the rectangle boundary."""
    return count_violations(edge_selection, grid) == 0 and reaches_boundary(edge_selection, grid)


def build_solution(edge_selection: List[int], grid: HananGrid) -> Solution:
    """Wrap an edge selection into a scored, validity-checked Solution."""
    return Solution(
        edge_selection=list(edge_selection),
        total_length=total_length(edge_selection, grid),
        is_valid=is_valid_partition(edge_selection, grid),
    )
