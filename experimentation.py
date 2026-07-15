"""
Experiment runner and figure generator.

Runs brute force, the comparison of every method, and the convergence curves,
writing all plots to `results/`.

Run with:  python experimentation.py
"""

import os

from hanan_grid import build_hanan_grid
from instances import random_instance
from exhaustive_search import solve_exhaustive
from branch_and_bound import solve_branch_and_bound
from simulated_annealing import solve_simulated_annealing
from genetic_algorithm import solve_genetic_algorithm
from plotting import plot_solution, plot_comparison, plot_convergence

OUTPUT_DIR = "results"


def report(name, solution, elapsed):
    """Print one result line."""
    print(f"  {name:14} length={solution.total_length:8.2f}  "
          f"valid={str(solution.is_valid):5}  time={elapsed:6.3f}s")


def winner(solutions):
    """Name of the shortest valid solution, or None if all are invalid."""
    valid = {name: s for name, s in solutions.items() if s.is_valid}
    if not valid:
        return None
    return min(valid, key=lambda name: valid[name].total_length)


def run_brute_force() -> None:
    """True brute force on the two smallest instances (2 and 3 points), saving a
    solution plot for each. Branch and bound is run on the same instances plus a
    4-point one to show it reaches sizes exhaustive search cannot."""
    print("=" * 70, "\nBRUTE FORCE (exact optimum)\n", "=" * 70, sep="")

    for num_points in [2, 3]:
        instance = random_instance(num_points, seed=42 + num_points)
        grid = build_hanan_grid(instance)
        solution, elapsed = solve_exhaustive(instance, grid)
        print(f"\n--- {num_points} points, {len(grid.internal_edges)} edges, "
              f"{2 ** len(grid.internal_edges):,} subsets ---")
        print(f"  Exhaustive    length={solution.total_length:.2f}  "
              f"valid={solution.is_valid}  time={elapsed:.2f}s")
        plot_solution(instance, grid, solution,
                      title=f"Brute-force optimum ({num_points} points)",
                      save_path=f"{OUTPUT_DIR}/bruteforce_{num_points}pts.png", show=False)

    print("\nBranch and bound (pruned exact search):")
    for num_points in [2, 3, 4]:
        instance = random_instance(num_points, seed=42 + num_points)
        grid = build_hanan_grid(instance)
        solution, elapsed = solve_branch_and_bound(instance, grid)
        print(f"  {num_points} points ({len(grid.internal_edges)} edges): "
              f"length={solution.total_length:.2f}  valid={solution.is_valid}  time={elapsed:.3f}s")
        if num_points == 4:
            plot_solution(instance, grid, solution,
                          title="Branch-and-bound optimum (4 points)",
                          save_path=f"{OUTPUT_DIR}/branch_and_bound_4pts.png", show=False)


def run_comparison() -> None:
    """Compare every method on the small instances (against the exact optimum) and
    on larger ones (heuristics only), saving a comparison plot per instance."""
    print("\n" + "=" * 70, "\nCOMPARISON\n", "=" * 70, sep="")
    for num_points in [2, 3, 8, 15]:
        instance = random_instance(num_points, seed=42 + num_points)
        grid = build_hanan_grid(instance)
        print(f"\n--- {num_points} points, {len(grid.internal_edges)} edges ---")

        solutions = {}

        if num_points <= 4:
            solution, elapsed = solve_branch_and_bound(instance, grid)
            report("Exact (B&B)", solution, elapsed)
            solutions["Exact (B&B)"] = solution

        solution, elapsed, _ = solve_simulated_annealing(
            instance, grid, max_iterations=20000, seed=123)
        report("SA", solution, elapsed)
        solutions["SA"] = solution

        solution, elapsed, _ = solve_genetic_algorithm(
            instance, grid, population_size=80, num_generations=200, seed=123)
        report("GA", solution, elapsed)
        solutions["GA"] = solution

        print(f"  Winner: {winner(solutions) or 'NONE (all invalid)'}")
        plot_comparison(instance, grid, solutions,
                        save_path=f"{OUTPUT_DIR}/comparison_{num_points}pts.png", show=False)


def run_convergence() -> None:
    """Convergence curves for SA and GA on a medium instance."""
    print("\n" + "=" * 70, "\nCONVERGENCE\n", "=" * 70, sep="")
    instance = random_instance(12, seed=112)
    grid = build_hanan_grid(instance)
    _, _, sa_stats = solve_simulated_annealing(instance, grid, max_iterations=20000, seed=7)
    _, _, ga_stats = solve_genetic_algorithm(instance, grid, population_size=100,
                                             num_generations=300, seed=7)
    plot_convergence(sa_stats=sa_stats, ga_stats=ga_stats,
                     save_path=f"{OUTPUT_DIR}/convergence.png", show=False)
    print(f"  Saved {OUTPUT_DIR}/convergence.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_brute_force()
    run_comparison()
    run_convergence()
    print(f"\nAll figures written to {OUTPUT_DIR}/")
