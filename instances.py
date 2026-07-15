"""
Test instance generators.

Because instances with distinct point coordinates are the hardest (no cut can be
shared between two points), we provide three families of increasing structure:
fully random, clustered, and coordinate-sharing.
"""

import random
from typing import Optional

from hanan_grid import Point, Rectangle, ProblemInstance


def random_instance(num_points: int, width: float = 100.0, height: float = 100.0,
                    seed: Optional[int] = None) -> ProblemInstance:
    """Random points with distinct x and distinct y coordinates."""
    if seed is not None:
        random.seed(seed)

    rectangle = Rectangle(0.0, 0.0, width, height)
    xs = random.sample(range(1, int(width)), num_points)
    ys = random.sample(range(1, int(height)), num_points)
    points = [Point(float(x), float(y)) for x, y in zip(xs, ys)]
    return ProblemInstance(rectangle, points)


def clustered_instance(num_points: int, num_clusters: int = 3,
                       width: float = 100.0, height: float = 100.0,
                       seed: Optional[int] = None) -> ProblemInstance:
    """Points grouped around a few cluster centers, so some share coordinates."""
    if seed is not None:
        random.seed(seed)

    rectangle = Rectangle(0.0, 0.0, width, height)
    cx_centers = random.sample(range(20, int(width) - 20), num_clusters)
    cy_centers = random.sample(range(20, int(height) - 20), num_clusters)

    points = []
    used = set()
    for cluster_idx in range(num_clusters):
        count = num_points // num_clusters + (1 if cluster_idx < num_points % num_clusters else 0)
        cx, cy = cx_centers[cluster_idx], cy_centers[cluster_idx]
        for _ in range(count):
            for _attempt in range(1000):
                x = max(1, min(int(width) - 1, cx + random.randint(-10, 10)))
                y = max(1, min(int(height) - 1, cy + random.randint(-10, 10)))
                if (x, y) not in used:
                    used.add((x, y))
                    points.append(Point(float(x), float(y)))
                    break

    return ProblemInstance(rectangle, points)


def shared_coordinates_instance(num_points: int, sharing_ratio: float = 0.3,
                                width: float = 100.0, height: float = 100.0,
                                seed: Optional[int] = None) -> ProblemInstance:
    """Points drawn from a limited pool of coordinates, so many share a row or
    column. `sharing_ratio` controls how small that pool is."""
    if seed is not None:
        random.seed(seed)

    rectangle = Rectangle(0.0, 0.0, width, height)
    num_unique = max(2, int(num_points * (1 - sharing_ratio)))
    available_x = random.sample(range(1, int(width)), min(num_unique, int(width) - 1))
    available_y = random.sample(range(1, int(height)), min(num_unique, int(height) - 1))

    points = []
    used = set()
    for _ in range(num_points):
        for _attempt in range(1000):
            x, y = random.choice(available_x), random.choice(available_y)
            if (x, y) not in used:
                used.add((x, y))
                points.append(Point(float(x), float(y)))
                break

    return ProblemInstance(rectangle, points)
