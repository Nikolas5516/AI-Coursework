"""
simulated_annealing.py
======================
Implementarea algoritmului Simulated Annealing (SA) pentru:
  - Problema rucsacului (knapsack)
  - Problema comis-voiajorului (TSP)

Schema de răcire: T(k+1) = alpha * T(k)
"""

import numpy as np
import time
from utils import (
    generate_valid_solution_rucsac, fitness_rucsac, random_neighbor_rucsac,
    generate_solution_TSP, fitness_TSP, two_swap, two_opt
)


# =============================================================================
# SA - RUCSAC
# =============================================================================

def simulated_annealing_rucsac(
    t: float,
    alpha: float,
    t_min: float,
    max_iterations: int,
    objects: list,
    max_capacity: int
) -> "tuple[list, int, list]":
    """
    Simulated Annealing pentru problema rucsacului.

    Pasii algoritmului:
    1. Genereaza o solutie valida aleatoare c.
    2. Cat timp t > t_min:
       a. Repeta de max_iterations ori:
          - Genereaza un vecin aleator x (flip un bit)
          - Calculeaza delta = fc - fx  (negativ = x e mai bun)
          - Daca delta < 0 (x mai bun): accepta x
          - Altfel: accepta x cu probabilitate exp(-delta/t)
       b. Reduce temperatura: t = t * alpha
    3. Returneaza cea mai buna solutie gasita.

    INPUT
    -----
    t             - temperatura initiala (ex: 1000)
    alpha         - coeficient racire (ex: 0.99), subunitar aproape de 1
    t_min         - temperatura minima de oprire (ex: 0.001)
    max_iterations- iteratii per nivel de temperatura
    objects       - lista de tupluri (greutate, valoare)
    max_capacity  - capacitatea maxima W

    OUTPUT
    ------
    c             - cea mai buna solutie gasita
    fc            - fitness-ul acesteia
    found_solutions - istoricul fitness-ului
    """
    # Pas 1: solutie initiala
    c = generate_valid_solution_rucsac(len(objects), objects, max_capacity)
    fc = fitness_rucsac(objects, c, max_capacity)
    best_c = c[:]
    best_fc = fc
    found_solutions = [fc]

    # Pas 2: bucla de racire
    while t > t_min:
        k = 0
        while k < max_iterations:
            x = random_neighbor_rucsac(c)
            fx = fitness_rucsac(objects, x, max_capacity)

            if fx == -1:  # solutie invalida, ignoram
                k += 1
                continue

            delta = fc - fx  # negativ daca x e mai bun (vrem maximizare)

            if delta < 0:
                # x e mai bun -> acceptam sigur
                c = x[:]
                fc = fx
            elif np.random.random() < np.exp(-delta / t):
                # x e mai slab dar acceptam cu probabilitate
                c = x[:]
                fc = fx

            if fc > best_fc:
                best_c = c[:]
                best_fc = fc

            found_solutions.append(best_fc)
            k += 1

        # Pas 2b: racire
        t = t * alpha

    return best_c, best_fc, found_solutions


def sa_rucsac_n_times(
    n: int,
    t: float,
    alpha: float,
    t_min: float,
    max_iterations: int,
    objects: list,
    max_capacity: int
) -> "tuple[list, float]":
    """Ruleaza SA pentru rucsac de n ori si returneaza statistici."""
    best_solutions = []
    total_time = 0.0
    for _ in range(n):
        start = time.time()
        sol, fit, _ = simulated_annealing_rucsac(t, alpha, t_min, max_iterations, objects, max_capacity)
        total_time += time.time() - start
        best_solutions.append((sol, fit))
    return best_solutions, total_time / n


# =============================================================================
# SA - TSP
# =============================================================================

def simulated_annealing_TSP(
    t: float,
    alpha: float,
    t_min: float,
    max_iterations: int,
    dm: np.ndarray,
    dimension: int,
    use_two_opt: bool = False
) -> "tuple[list, int, list]":
    """
    Simulated Annealing pentru TSP.

    Diferenta fata de rucsac:
    - Solutia = permutare de orase (nu vector binar)
    - Fitness = distanta totala (vrem MINIMIZARE, nu maximizare)
    - delta = fx - fc  (pozitiv = x e mai rau; acceptam cu probabilitate)
    - Vecinatate: 2-swap (default) sau 2-opt (parametrizabil)

    INPUT
    -----
    t             - temperatura initiala
    alpha         - coeficient racire
    t_min         - temperatura minima
    max_iterations- iteratii per nivel temperatura
    dm            - matricea de distante
    dimension     - numarul de orase
    use_two_opt   - True = foloseste 2-opt, False = foloseste 2-swap

    OUTPUT
    ------
    best_c        - cea mai buna solutie (distanta minima)
    best_fc       - distanta acesteia
    found_solutions - istoricul distantei minime
    """
    # Pas 1: solutie initiala aleatoare
    c = generate_solution_TSP(dimension)
    fc = fitness_TSP(c, dm)
    best_c = c[:]
    best_fc = fc
    found_solutions = [fc]

    # Pas 2: bucla de racire
    while t > t_min:
        k = 0
        while k < max_iterations:
            # Genereaza vecin: 2-swap sau 2-opt
            if use_two_opt:
                i = np.random.randint(dimension - 1)
                j = np.random.randint(i + 1, dimension)
                x = two_opt(c, i, j)
            else:
                x = two_swap(c, dimension)

            fx = fitness_TSP(x, dm)
            delta = fx - fc  # pozitiv = x e mai rau (minimizare)

            if delta < 0:
                # x e mai bun (distanta mai mica) -> acceptam sigur
                c = x[:]
                fc = fx
            elif t > 0 and np.random.random() < np.exp(-delta / t):
                # x e mai rau dar acceptam cu probabilitate
                c = x[:]
                fc = fx

            if fc < best_fc:
                best_c = c[:]
                best_fc = fc

            found_solutions.append(best_fc)
            k += 1

        t = t * alpha

    return best_c, best_fc, found_solutions


def sa_TSP_n_times(
    n: int,
    t: float,
    alpha: float,
    t_min: float,
    max_iterations: int,
    dm: np.ndarray,
    dimension: int,
    use_two_opt: bool = False
) -> "tuple[list, float]":
    """Ruleaza SA pentru TSP de n ori si returneaza statistici."""
    best_solutions = []
    total_time = 0.0
    for _ in range(n):
        start = time.time()
        sol, fit, _ = simulated_annealing_TSP(t, alpha, t_min, max_iterations, dm, dimension, use_two_opt)
        total_time += time.time() - start
        best_solutions.append((sol, fit))
    return best_solutions, total_time / n