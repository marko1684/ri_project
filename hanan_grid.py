"""
Geometric model and Hanan grid construction.

For the Minimum Partition of a Rectangle with Interior Points problem, the
optimal set of cutting segments always lies on the Hanan grid: the grid formed
by drawing a horizontal and a vertical line through every interior point and
clipping to the rectangle. Choosing a partition therefore reduces to picking a
subset of the internal grid edges, which is what every solver in this package
does.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __repr__(self):
        return f"({self.x}, {self.y})"


@dataclass
class Rectangle:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass
class ProblemInstance:
    rectangle: Rectangle
    points: List[Point]

    @property
    def num_points(self) -> int:
        return len(self.points)


@dataclass
class HananGrid:
    """
    Hanan grid built from a rectangle and its interior points.

    Vertices are the intersections of the horizontal and vertical lines through
    the points. Edges connecting adjacent vertices are split into internal edges
    (candidates a solver may select) and boundary edges (the rectangle perimeter,
    always present).
    """
    x_coords: List[float]
    y_coords: List[float]
    vertices: List[Point]
    internal_edges: List[Tuple[int, int]]
    edge_lengths: List[float]
    boundary_edges: List[Tuple[int, int]]
    vertex_to_internal_edges: Dict[int, List[int]]
    vertex_to_boundary_edges: Dict[int, List[int]]
    point_vertex_indices: List[int]
    is_boundary_vertex: List[bool]


def build_hanan_grid(instance: ProblemInstance) -> HananGrid:
    """Construct the Hanan grid for a problem instance."""
    rect = instance.rectangle

    x_coords = sorted(set([rect.x_min, rect.x_max] + [p.x for p in instance.points]))
    y_coords = sorted(set([rect.y_min, rect.y_max] + [p.y for p in instance.points]))
    num_x, num_y = len(x_coords), len(y_coords)

    vertex_index = {}
    vertices = []
    for y in y_coords:
        for x in x_coords:
            vertex_index[(x, y)] = len(vertices)
            vertices.append(Point(x, y))

    is_boundary_vertex = [
        v.x in (rect.x_min, rect.x_max) or v.y in (rect.y_min, rect.y_max)
        for v in vertices
    ]

    internal_edges = []
    edge_lengths = []
    boundary_edges = []

    def on_boundary(v1: Point, v2: Point) -> bool:
        if v1.y == v2.y and v1.y in (rect.y_min, rect.y_max):
            return True
        if v1.x == v2.x and v1.x in (rect.x_min, rect.x_max):
            return True
        return False

    def add_edge(v1_idx: int, v2_idx: int, length: float):
        if on_boundary(vertices[v1_idx], vertices[v2_idx]):
            boundary_edges.append((v1_idx, v2_idx))
        else:
            internal_edges.append((v1_idx, v2_idx))
            edge_lengths.append(length)

    for yi in range(num_y):
        for xi in range(num_x - 1):
            v1 = vertex_index[(x_coords[xi], y_coords[yi])]
            v2 = vertex_index[(x_coords[xi + 1], y_coords[yi])]
            add_edge(v1, v2, x_coords[xi + 1] - x_coords[xi])

    for xi in range(num_x):
        for yi in range(num_y - 1):
            v1 = vertex_index[(x_coords[xi], y_coords[yi])]
            v2 = vertex_index[(x_coords[xi], y_coords[yi + 1])]
            add_edge(v1, v2, y_coords[yi + 1] - y_coords[yi])

    vertex_to_internal_edges = defaultdict(list)
    for edge_idx, (v1, v2) in enumerate(internal_edges):
        vertex_to_internal_edges[v1].append(edge_idx)
        vertex_to_internal_edges[v2].append(edge_idx)

    vertex_to_boundary_edges = defaultdict(list)
    for edge_idx, (v1, v2) in enumerate(boundary_edges):
        vertex_to_boundary_edges[v1].append(edge_idx)
        vertex_to_boundary_edges[v2].append(edge_idx)

    point_vertex_indices = [vertex_index[(p.x, p.y)] for p in instance.points]

    return HananGrid(
        x_coords=x_coords,
        y_coords=y_coords,
        vertices=vertices,
        internal_edges=internal_edges,
        edge_lengths=edge_lengths,
        boundary_edges=boundary_edges,
        vertex_to_internal_edges=dict(vertex_to_internal_edges),
        vertex_to_boundary_edges=dict(vertex_to_boundary_edges),
        point_vertex_indices=point_vertex_indices,
        is_boundary_vertex=is_boundary_vertex,
    )
