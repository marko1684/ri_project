"""
Genetic algorithm over the Hanan grid edge selection.

A chromosome is the binary edge vector. Fitness is the negated penalized cost,
so shorter valid partitions score highest. The initial population mixes a few
greedy constructed individuals (valid solutions) with random ones (diversity).
Each generation keeps the elite, then fills the rest with tournament selected
parents, crossover (uniform or two point), and a line aware mutation that
occasionally flips a whole grid line.
"""

import random
import time
from typing import List, Optional, Tuple

from hanan_grid import ProblemInstance, HananGrid
from partition import (
    Solution, build_solution, total_length, is_valid_partition, penalized_cost,
)
from greedy import greedy_construction


def fitness(individual: List[int], grid: HananGrid) -> float:
    """Higher is better: the negated penalized cost."""
    return -penalized_cost(individual, grid)


def initial_population(grid: HananGrid, population_size: int) -> List[List[int]]:
    """A quarter greedy individuals (valid anchors), the rest random."""
    num_edges = len(grid.internal_edges)
    num_greedy = max(1, population_size // 4)
    greedy = greedy_construction(grid)
    population = [greedy[:] for _ in range(num_greedy)]
    population += [[random.randint(0, 1) for _ in range(num_edges)]
                  for _ in range(population_size - num_greedy)]
    return population


def tournament_selection(population: List[List[int]], fitnesses: List[float],
                         tournament_size: int = 3) -> List[int]:
    """Pick the fittest of `tournament_size` random individuals."""
    contenders = random.sample(range(len(population)), min(tournament_size, len(population)))
    return population[max(contenders, key=lambda i: fitnesses[i])][:]


def crossover(parent_a: List[int], parent_b: List[int]) -> Tuple[List[int], List[int]]:
    """Uniform crossover half the time, two point crossover otherwise."""
    n = len(parent_a)
    if random.random() < 0.5 or n <= 2:
        child_a, child_b = [], []
        for gene_a, gene_b in zip(parent_a, parent_b):
            if random.random() < 0.5:
                child_a.append(gene_a); child_b.append(gene_b)
            else:
                child_a.append(gene_b); child_b.append(gene_a)
        return child_a, child_b

    p1 = random.randint(0, n - 2)
    p2 = random.randint(p1 + 1, n - 1)
    return (parent_a[:p1] + parent_b[p1:p2] + parent_a[p2:],
            parent_b[:p1] + parent_a[p1:p2] + parent_b[p2:])


def mutate(individual: List[int], grid: HananGrid, mutation_rate: float) -> List[int]:
    """Bit flip mutation, occasionally preceded by flipping a whole grid line so
    the operator respects the geometry of the problem."""
    mutated = individual[:]

    if random.random() < mutation_rate * 5:
        horizontal = random.random() < 0.5
        coords = grid.y_coords[1:-1] if horizontal else grid.x_coords[1:-1]
        if coords:
            target = random.choice(coords)
            for i, (v1_idx, v2_idx) in enumerate(grid.internal_edges):
                v1, v2 = grid.vertices[v1_idx], grid.vertices[v2_idx]
                on_line = (v1.y == target and v2.y == target) if horizontal else \
                          (v1.x == target and v2.x == target)
                if on_line:
                    mutated[i] ^= 1

    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] ^= 1

    return mutated


def solve_genetic_algorithm(instance: ProblemInstance, grid: HananGrid,
                            population_size: int = 100,
                            num_generations: int = 500,
                            crossover_rate: float = 0.85,
                            mutation_rate: float = 0.03,
                            elitism_count: int = 5,
                            tournament_size: int = 3,
                            time_limit_seconds: Optional[float] = None,
                            seed: Optional[int] = None) -> Tuple[Solution, float, dict]:
    """Run the genetic algorithm. Returns best VALID solution, and stats with time."""
    if seed is not None:
        random.seed(seed)

    start = time.time()
    population = initial_population(grid, population_size)
    fitnesses = [fitness(ind, grid) for ind in population]

    best_valid = None
    best_valid_length = float("inf")
    best_cost_history = []
    avg_cost_history = []

    def record(generation_fitnesses):
        nonlocal best_valid, best_valid_length
        for ind in population:
            if is_valid_partition(ind, grid):
                length = total_length(ind, grid)
                if length < best_valid_length:
                    best_valid_length, best_valid = length, ind[:]
        best_cost_history.append(best_valid_length if best_valid else -max(generation_fitnesses))
        avg_cost_history.append(-sum(generation_fitnesses) / len(generation_fitnesses))

    record(fitnesses)

    generation = 0
    for generation in range(num_generations):
        if time_limit_seconds is not None and time.time() - start > time_limit_seconds:
            break

        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        new_population = [population[i][:] for i in ranked[:elitism_count]]

        while len(new_population) < population_size:
            parent_a = tournament_selection(population, fitnesses, tournament_size)
            parent_b = tournament_selection(population, fitnesses, tournament_size)
            if random.random() < crossover_rate:
                child_a, child_b = crossover(parent_a, parent_b)
            else:
                child_a, child_b = parent_a[:], parent_b[:]
            new_population.append(mutate(child_a, grid, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutate(child_b, grid, mutation_rate))

        population = new_population
        fitnesses = [fitness(ind, grid) for ind in population]
        record(fitnesses)

    selection = best_valid if best_valid is not None else \
        population[max(range(len(population)), key=lambda i: fitnesses[i])]
    stats = {
        "generations_completed": generation + 1,
        "best_cost_history": best_cost_history,
        "avg_cost_history": avg_cost_history,
    }
    return build_solution(selection, grid), time.time() - start, stats
