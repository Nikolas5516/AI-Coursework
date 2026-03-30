"""
utils.py
========
Funcții utilitare comune: încărcare date, distanțe, vizualizare.
"""

import numpy as np
import matplotlib.pyplot as plt
import math


# =============================================================================
# RUCSAC - încărcare date
# =============================================================================

def load_data_rucsac(file_name: str) -> "tuple[list[tuple[int, int]], int]":
    """
    Citeste datele din fisierul file_name pentru problema rucsacului.
    Format: prima linie = n, urmatoarele n linii = index greutate valoare,
            ultima linie = capacitate maxima.
    """
    weights_and_values = []
    with open(file_name) as f:
        nr_obj = int(f.readline())
        for i in range(nr_obj):
            line = f.readline().split(" ")
            line = [x for x in line if x != ""]
            weights_and_values.append(tuple(int(x) for x in line[1:]))
        max_capacity = int(f.readline())
    return weights_and_values, max_capacity


# =============================================================================
# RUCSAC - fitness și generare
# =============================================================================

def is_valid_rucsac(objects: list, sol: list, max_capacity: int) -> bool:
    total_weight = sum(objects[i][0] * sol[i] for i in range(len(sol)))
    return total_weight <= max_capacity


def fitness_rucsac(objects: list, sol: list, max_capacity: int) -> int:
    if not is_valid_rucsac(objects, sol, max_capacity):
        return -1
    return sum(objects[i][1] * sol[i] for i in range(len(sol)))


def generate_solution_rucsac(n: int) -> list:
    return list(np.random.randint(2, size=n))


def generate_valid_solution_rucsac(n: int, objects: list, max_capacity: int) -> list:
    sol = []
    while not sol or not is_valid_rucsac(objects, sol, max_capacity):
        sol = generate_solution_rucsac(n)
    return sol


def random_neighbor_rucsac(sol: list) -> list:
    """Vecin aleator: flip un bit."""
    x = sol[:]
    index = np.random.randint(len(x))
    x[index] = 1 - x[index]
    return x


# =============================================================================
# TSP - încărcare date
# =============================================================================

def read_from_file_TSP(name: str) -> "tuple[list[tuple], int]":
    """
    Citeste datele din fisierul .tsp (format TSPLIB).
    Returneaza lista de locatii (index, x, y) si dimensiunea.
    """
    locations = []
    dimension = 0
    with open(name, 'r') as file:
        for i in range(6):
            line = file.readline().replace("\n", "").split(" ")
            if len(line) == 2 and line[0] == "DIMENSION:":
                dimension = int(line[1])
        for i in range(dimension):
            line = file.readline().replace("\n", "").split(" ")
            line = [x for x in line if x != ""]
            locations.append(tuple(int(x) for x in line[:]))
    return locations, dimension


def read_from_file_TSP_opt(name: str) -> list:
    """
    Citeste solutia optima din fisierul .opt.tour (format TSPLIB).
    """
    solution = []
    dimension = 0
    with open(name, 'r') as file:
        for i in range(5):
            line = file.readline().replace("\n", "").split(" ")
            if len(line) == 3 and line[0] == "DIMENSION":
                dimension = int(line[2])
        for i in range(dimension):
            line = file.readline().replace("\n", "")
            solution.append(int(line) - 1)
    return solution


def distance_matrix_TSP(locations: list) -> np.ndarray:
    """
    Calculeaza matricea de distante euclidiene pentru toate perechile de orase.
    """
    length = len(locations)
    dm = np.zeros([length, length], dtype=int)
    for i in range(length):
        for j in range(i + 1, length):
            xd = locations[i][1] - locations[j][1]
            yd = locations[i][2] - locations[j][2]
            dist = int(math.sqrt(xd ** 2 + yd ** 2))
            dm[i][j] = dist
            dm[j][i] = dist
    return dm


# =============================================================================
# TSP - solutie si fitness
# =============================================================================

def generate_solution_TSP(dimension: int) -> list:
    """Genereaza o permutare aleatoare a oraselor."""
    import random
    sol = list(range(dimension))
    random.shuffle(sol)
    return sol


def fitness_TSP(solution: list, dm: np.ndarray) -> int:
    """Calculeaza lungimea totala a drumului (inclusiv intoarcerea la start)."""
    distance = 0
    n = len(solution)
    for i in range(n):
        distance += dm[solution[i]][solution[(i + 1) % n]]
    return distance


# =============================================================================
# TSP - operatori de vecinătate
# =============================================================================

def two_swap(solution: list, dimension: int) -> list:
    """
    2-swap: interschimba doua orase aleatoare din solutie.
    """
    x = solution[:]
    i = np.random.randint(dimension)
    j = np.random.randint(dimension)
    while j == i:
        j = np.random.randint(dimension)
    x[i], x[j] = x[j], x[i]
    return x


def two_opt(solution: list, i: int, j: int) -> list:
    """
    2-opt: oglindeste sub-sirul dintre pozitiile i si j (inclusiv).
    Ex: [A, B, C, D, E] cu i=1, j=3 => [A, D, C, B, E]
    """
    x = solution[:]
    x[i:j + 1] = reversed(x[i:j + 1])
    return x


def get_all_two_opt_neighbors(solution: list) -> "list[tuple[list, int, int]]":
    """
    Genereaza toti vecinii 2-opt ai unei solutii.
    Returneaza lista de (solutie_vecina, i, j) pentru a putea actualiza memoria tabu.
    """
    neighbors = []
    n = len(solution)
    for i in range(n - 1):
        for j in range(i + 1, n):
            neighbor = two_opt(solution, i, j)
            neighbors.append((neighbor, i, j))
    return neighbors


def get_all_two_swap_neighbors(solution: list) -> "list[tuple[list, int, int]]":
    """
    Genereaza toti vecinii 2-swap ai unei solutii.
    Returneaza lista de (solutie_vecina, i, j).
    """
    neighbors = []
    n = len(solution)
    for i in range(n):
        for j in range(i + 1, n):
            x = solution[:]
            x[i], x[j] = x[j], x[i]
            neighbors.append((x, i, j))
    return neighbors


# =============================================================================
# VIZUALIZARE
# =============================================================================

def plot_fitness_history(history: list, title: str = "Evolutie fitness", ylabel: str = "Fitness"):
    """Afiseaza graficul evolutiei fitness-ului de-a lungul iteratiilor."""
    plt.figure(figsize=(10, 4))
    plt.plot(history, color="steelblue", linewidth=1)
    plt.title(title)
    plt.xlabel("Iteratie")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_tsp_route(locations: list, solution: list, title: str = "Ruta TSP"):
    """Afiseaza ruta TSP pe un grafic 2D."""
    coords = [(locations[i][1], locations[i][2]) for i in solution]
    coords.append(coords[0])  # inchide ruta
    xs, ys = zip(*coords)

    plt.figure(figsize=(8, 8))
    plt.plot(xs, ys, 'b-', linewidth=0.8, alpha=0.7)
    plt.scatter([l[1] for l in locations], [l[2] for l in locations],
                c='red', s=20, zorder=5)
    plt.title(title)
    plt.tight_layout()
    plt.show()