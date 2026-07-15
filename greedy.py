"""
Greedy construction heuristic.

Starting from the full grid (all internal edges selected, which is always a
valid partition), remove edges from longest to shortest and keep each removal
only while the partition stays valid. This yields a cheap valid solution.
"""

from typing import List

from hanan_grid import HananGrid
from partition import is_valid_partition


def greedy_construction(grid: HananGrid) -> List[int]:
    """Build a valid selection by greedily dropping the longest removable edges."""
    selection = [1] * len(grid.internal_edges)
    order = sorted(range(len(grid.internal_edges)),
                   key=lambda i: grid.edge_lengths[i], reverse=True)

    for edge_idx in order:
        selection[edge_idx] = 0
        if not is_valid_partition(selection, grid):
            selection[edge_idx] = 1

    return selection
