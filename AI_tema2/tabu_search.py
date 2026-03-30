"""
tabu_search.py
==============
Implementarea algoritmului Tabu Search (TS) pentru:
  - Problema rucsacului (knapsack)
  - Problema comis-voiajorului (TSP) cu 2-swap si 2-opt parametrizabile

Pseudocod general:
    c = initSolution()
    best = c
    M = initMemory()          # lista/matricea tabu
    while stop-criterion not met:
        x = bestNeighbourNonTabu(c, M)
        updateMemory(M)
        c = x
        update best
    return best
"""

import numpy as np
import time
from utils import (
    generate_valid_solution_rucsac, fitness_rucsac,
    generate_solution_TSP, fitness_TSP,
    get_all_two_swap_neighbors, get_all_two_opt_neighbors
)


# =============================================================================
# TS - RUCSAC
# =============================================================================

def tabu_search_rucsac(
    max_iterations: int,
    tabu_tenure: int,
    objects: list,
    max_capacity: int
) -> "tuple[list, int, list]":
    """
    Tabu Search pentru problema rucsacului.

    Memoria tabu retine, pentru fiecare pozitie (bit) din solutie,
    numarul de iteratii cat timp nu este permis sa fie modificata (flip).

    Pasii algoritmului:
    1. Genereaza solutie initiala valida c; best = c.
    2. Initializeaza memoria tabu: tabu[i] = 0 pentru toti i.
    3. Cat timp iteratia curenta < max_iterations:
       a. Genereaza toti vecinii (flip fiecare bit ne-tabu).
       b. Alege cel mai bun vecin valid (chiar daca e mai slab decat c).
       c. Actualizeaza memoria: tabu[bit_modificat] = tabu_tenure.
       d. c = cel mai bun vecin; actualizeaza best daca e cazul.
       e. Decrementeaza toti contoarele tabu > 0.
    4. Returneaza best.

    INPUT
    -----
    max_iterations - numarul maxim de iteratii (criteriu de oprire)
    tabu_tenure    - numarul de iteratii cat un bit ramane tabu
    objects        - lista de tupluri (greutate, valoare)
    max_capacity   - capacitatea maxima W

    OUTPUT
    ------
    best_c         - cea mai buna solutie gasita
    best_fc        - fitness-ul acesteia
    history        - istoricul fitness-ului best la fiecare iteratie
    """
    n = len(objects)

    # Pas 1: initializare
    c = generate_valid_solution_rucsac(n, objects, max_capacity)
    fc = fitness_rucsac(objects, c, max_capacity)
    best_c = c[:]
    best_fc = fc
    history = [best_fc]

    # Pas 2: memoria tabu (un contor per bit)
    tabu = [0] * n

    # Pas 3: bucla principala
    for iteration in range(max_iterations):
        best_neighbor = None
        best_neighbor_fitness = -1
        best_neighbor_idx = -1

        # Pas 3a: evalueaza toti vecinii ne-tabu
        for i in range(n):
            if tabu[i] > 0:
                continue  # bit-ul i este tabu, sarim
            neighbor = c[:]
            neighbor[i] = 1 - neighbor[i]  # flip bit i
            fn = fitness_rucsac(objects, neighbor, max_capacity)
            if fn == -1:
                continue  # solutie invalida

            # Pas 3b: retine cel mai bun vecin ne-tabu
            if best_neighbor is None or fn > best_neighbor_fitness:
                best_neighbor = neighbor[:]
                best_neighbor_fitness = fn
                best_neighbor_idx = i

        if best_neighbor is None:
            # toti vecinii sunt tabu sau invalizi - continuam
            history.append(best_fc)
            for i in range(n):
                if tabu[i] > 0:
                    tabu[i] -= 1
            continue

        # Pas 3c: actualizeaza memoria tabu
        tabu[best_neighbor_idx] = tabu_tenure

        # Pas 3d: muta la cel mai bun vecin
        c = best_neighbor[:]
        fc = best_neighbor_fitness

        if fc > best_fc:
            best_c = c[:]
            best_fc = fc

        history.append(best_fc)

        # Pas 3e: decrementeaza contoarele tabu
        for i in range(n):
            if tabu[i] > 0:
                tabu[i] -= 1

    return best_c, best_fc, history


def ts_rucsac_n_times(
    n_runs: int,
    max_iterations: int,
    tabu_tenure: int,
    objects: list,
    max_capacity: int
) -> "tuple[list, float]":
    """Ruleaza TS pentru rucsac de n_runs ori si returneaza statistici."""
    best_solutions = []
    total_time = 0.0
    for _ in range(n_runs):
        start = time.time()
        sol, fit, _ = tabu_search_rucsac(max_iterations, tabu_tenure, objects, max_capacity)
        total_time += time.time() - start
        best_solutions.append((sol, fit))
    return best_solutions, total_time / n_runs


# =============================================================================
# TS - TSP
# =============================================================================

def tabu_search_TSP(
    max_iterations: int,
    tabu_tenure: int,
    dm: np.ndarray,
    dimension: int,
    use_two_opt: bool = False
) -> "tuple[list, int, list]":
    """
    Tabu Search pentru TSP cu 2-swap sau 2-opt (parametrizabil).

    Memoria tabu: o matrice tabu[i][j] retine numarul de iteratii cat timp
    miscarea care implica orasele i si j (swap sau opt) este interzisa.

    Pasii algoritmului:
    1. Genereaza solutie initiala aleatoare c; best = c.
    2. Initializeaza matricea tabu: tabu[i][j] = 0 pentru toti i, j.
    3. Cat timp iteratia < max_iterations:
       a. Genereaza toti vecinii (2-swap sau 2-opt).
       b. Alege cel mai bun vecin a carui miscare (i,j) NU este tabu.
          (Exceptie aspiratie: daca vecinul e mai bun decat best global,
           il acceptam chiar daca e tabu.)
       c. Actualizeaza memoria: tabu[i][j] = tabu[j][i] = tabu_tenure.
       d. c = cel mai bun vecin; actualizeaza best daca e cazul.
       e. Decrementeaza toate contoarele tabu > 0.
    4. Returneaza best.

    INPUT
    -----
    max_iterations - numarul maxim de iteratii
    tabu_tenure    - numarul de iteratii cat o miscare ramane tabu
    dm             - matricea de distante
    dimension      - numarul de orase
    use_two_opt    - True = 2-opt, False = 2-swap

    OUTPUT
    ------
    best_c         - cea mai buna solutie (distanta minima)
    best_fc        - distanta acesteia
    history        - istoricul distantei minime
    """
    # Pas 1: initializare
    c = generate_solution_TSP(dimension)
    fc = fitness_TSP(c, dm)
    best_c = c[:]
    best_fc = fc
    history = [best_fc]

    # Pas 2: matricea tabu (dimension x dimension)
    tabu = np.zeros((dimension, dimension), dtype=int)

    # Pas 3: bucla principala
    for iteration in range(max_iterations):
        # Pas 3a: genereaza toti vecinii
        if use_two_opt:
            neighbors = get_all_two_opt_neighbors(c)
        else:
            neighbors = get_all_two_swap_neighbors(c)

        # Pas 3b: alege cel mai bun vecin ne-tabu (cu criteriu de aspiratie)
        best_neighbor = None
        best_neighbor_fitness = float('inf')
        best_move = (-1, -1)

        for neighbor, i, j in neighbors:
            fn = fitness_TSP(neighbor, dm)
            is_tabu = tabu[i][j] > 0

            # Criteriu de aspiratie: acceptam chiar daca e tabu daca bate best global
            if not is_tabu or fn < best_fc:
                if fn < best_neighbor_fitness:
                    best_neighbor = neighbor[:]
                    best_neighbor_fitness = fn
                    best_move = (i, j)

        if best_neighbor is None:
            history.append(best_fc)
            tabu = np.maximum(tabu - 1, 0)
            continue

        # Pas 3c: actualizeaza memoria tabu
        bi, bj = best_move
        tabu[bi][bj] = tabu_tenure
        tabu[bj][bi] = tabu_tenure

        # Pas 3d: muta
        c = best_neighbor[:]
        fc = best_neighbor_fitness

        if fc < best_fc:
            best_c = c[:]
            best_fc = fc

        history.append(best_fc)

        # Pas 3e: decrementeaza toti contoarele
        tabu = np.maximum(tabu - 1, 0)

    return best_c, best_fc, history


def ts_TSP_n_times(
    n_runs: int,
    max_iterations: int,
    tabu_tenure: int,
    dm: np.ndarray,
    dimension: int,
    use_two_opt: bool = False
) -> "tuple[list, float]":
    """Ruleaza TS pentru TSP de n_runs ori si returneaza statistici."""
    best_solutions = []
    total_time = 0.0
    for _ in range(n_runs):
        start = time.time()
        sol, fit, _ = tabu_search_TSP(max_iterations, tabu_tenure, dm, dimension, use_two_opt)
        total_time += time.time() - start
        best_solutions.append((sol, fit))
    return best_solutions, total_time / n_runs