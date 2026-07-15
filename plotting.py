"""
Plotting of instances, partitions, and convergence curves.

Horizontal segments are drawn blue and vertical segments red over a copy
of the full Hanan grid, with the interior points on top.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Dict

from hanan_grid import ProblemInstance, HananGrid
from partition import Solution

HORIZONTAL_COLOR = "#2196F3"
VERTICAL_COLOR = "#F44336"


def _draw_partition(ax, instance: ProblemInstance, grid: HananGrid, solution: Solution,
                    title: str):
    """Draw one partition (rectangle, faint grid, selected edges, points) on `ax`."""
    rect = instance.rectangle
    ax.add_patch(patches.Rectangle(
        (rect.x_min, rect.y_min), rect.width, rect.height,
        linewidth=2.5, edgecolor="black", facecolor="lightyellow",
    ))

    for x in grid.x_coords[1:-1]:
        ax.axvline(x=x, color="gray", linewidth=0.3, alpha=0.3, linestyle="--")
    for y in grid.y_coords[1:-1]:
        ax.axhline(y=y, color="gray", linewidth=0.3, alpha=0.3, linestyle="--")

    for i, selected in enumerate(solution.edge_selection):
        if not selected:
            continue
        v1, v2 = (grid.vertices[k] for k in grid.internal_edges[i])
        color = HORIZONTAL_COLOR if abs(v1.y - v2.y) < 1e-9 else VERTICAL_COLOR
        ax.plot([v1.x, v2.x], [v1.y, v2.y], color=color, linewidth=2.0,
                alpha=0.8, solid_capstyle="round")

    ax.scatter([p.x for p in instance.points], [p.y for p in instance.points],
               color="black", s=45, zorder=5, edgecolors="white", linewidths=0.5)

    margin = max(rect.width, rect.height) * 0.05
    ax.set_xlim(rect.x_min - margin, rect.x_max + margin)
    ax.set_ylim(rect.y_min - margin, rect.y_max + margin)
    ax.set_aspect("equal")
    status = "VALID" if solution.is_valid else "INVALID"
    ax.set_title(f"{title}\nLength: {solution.total_length:.2f} [{status}]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _save_or_show(save_path: Optional[str], show: bool):
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_solution(instance: ProblemInstance, grid: HananGrid, solution: Solution,
                  title: str = "Partition", save_path: Optional[str] = None,
                  show: bool = True):
    """Plot a single partition."""
    _, ax = plt.subplots(figsize=(8, 8))
    _draw_partition(ax, instance, grid, solution, title)
    ax.plot([], [], color=HORIZONTAL_COLOR, linewidth=2.0, label="Horizontal segments")
    ax.plot([], [], color=VERTICAL_COLOR, linewidth=2.0, label="Vertical segments")
    ax.scatter([], [], color="black", s=45, label="Interior points")
    ax.legend(loc="upper right")
    _save_or_show(save_path, show)


def plot_comparison(instance: ProblemInstance, grid: HananGrid,
                    solutions: Dict[str, Solution],
                    save_path: Optional[str] = None, show: bool = True):
    """Plot several partitions of the same instance side by side."""
    _, axes = plt.subplots(1, len(solutions), figsize=(7 * len(solutions), 7))
    if len(solutions) == 1:
        axes = [axes]
    for ax, (name, solution) in zip(axes, solutions.items()):
        _draw_partition(ax, instance, grid, solution, name)
    _save_or_show(save_path, show)


def plot_convergence(sa_stats: Optional[dict] = None, ga_stats: Optional[dict] = None,
                     save_path: Optional[str] = None, show: bool = True):
    """Plot convergence curves for simulated annealing and/or the genetic algorithm."""
    _, ax = plt.subplots(figsize=(10, 6))

    if sa_stats and sa_stats.get("cost_history"):
        history = sa_stats["cost_history"]
        ax.plot(range(len(history)), history, "b-", label="Simulated Annealing", alpha=0.8)
    if ga_stats and ga_stats.get("best_cost_history"):
        history = ga_stats["best_cost_history"]
        ax.plot(range(len(history)), history, "r-", label="Genetic Algorithm (best)", alpha=0.8)
    if ga_stats and ga_stats.get("avg_cost_history"):
        history = ga_stats["avg_cost_history"]
        ax.plot(range(len(history)), history, "r--", label="GA (population avg)", alpha=0.4)

    ax.set_xlabel("Sample (SA every 500 iters) / Generation (GA)")
    ax.set_ylabel("Best valid total length")
    ax.set_title("Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_show(save_path, show)
