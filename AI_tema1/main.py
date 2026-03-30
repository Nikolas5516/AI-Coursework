"""
STEEPEST ASCENT HILL-CLIMBING (SAHC) - Problema Rucsacului
============================================================
Implementare SAHC pentru problema rucsacului.

Algoritmul:
    1. Se selecteaza un punct aleator c (current hilltop) în spatiul de cautare.
    2. Se determina TOATE punctele x din vecinatatea lui c: x apartine N(c)
    3. Daca oricare x ∈ N(c) are un fitness mai bun decât c, atunci c = x,
       unde x are cea mai buna valoare eval(x).
    4. Daca niciun punct x ∈ N(c) nu are un fitness mai bun decat c,
       se salveaza c si se trece la pasul 1.
       Altfel, se trece la pasul 2 cu noul c.
    5. Dupa un numar maxim de evaluari, se returneaza cel mai bun c (hilltop).

Diferenta fata de RHC:
    - RHC: evalueaza un singur vecin aleator per iteratie
    - SAHC: evalueaza TOTI vecinii si alege cel mai bun
"""

import numpy as np
import time


def load_data(file_name: str) -> "tuple[list[tuple[int, int]], int]":
    """
    Citește datele problemei rucsacului dintr-un fișier text.

    Formatul fișierului:
        - linia 1: numărul de obiecte (n)
        - liniile 2..n+1: index greutate valoare (separate prin spații)
        - ultima linie: capacitatea maxima W
    """
    weights_and_values = []
    with open(file_name) as f:
        nr_obj = int(f.readline())
        for i in range(nr_obj):
            line = f.readline()
            line = line.split(" ")
            line = [x for x in line if x != ""]
            weights_and_values.append(tuple(int(x) for x in line[1:]))
        max_capacity = int(f.readline())
    return weights_and_values, max_capacity


def is_valid(objects: list, sol: list, max_capacity: int) -> bool:
    """
    Verifica daca o solutie respecta constrangerea de greutate.
    """
    total_weight = sum(obj[0] * s for obj, s in zip(objects, sol))
    return total_weight <= max_capacity


def fitness(objects: list, sol: list, max_capacity: int) -> int:
    """
    Evalueaza o solutie si returneaza valoarea totala a obiectelor selectate.
    """
    value = 0
    total_weight = 0
    for (weight, val), selected in zip(objects, sol):
        if selected == 1:
            value += val
            total_weight += weight

    if total_weight > max_capacity:
        return -1

    return value


def generate_solution(n: int) -> list:
    """
    Genereaza o solutie aleatoare.
    """
    return list(np.random.randint(2, size=n))


def generate_valid_solution(n: int, objects: list, max_capacity: int) -> list:
    """
    Genereaza o solutie valida aleatoare (respecta constrangerea de greutate).
    Incearca solutii aleatoare pana gaseste una valida.
    """
    stop = False
    sol = []
    while not stop:
        sol = generate_solution(n)
        stop = is_valid(objects, sol, max_capacity)
    return sol



def get_all_neighbors(sol: list) -> list:
    """
    Genereaza TOTI vecinii unei soluții.
    """
    neighbors = []
    for i in range(len(sol)):
        neighbor = sol[:]          # copie a soluției curente
        neighbor[i] = 1 - neighbor[i]  # flip bit la poziția i
        neighbors.append(neighbor)
    return neighbors


def steepest_ascent_hill_climbing(
    max_evaluations: int,
    objects: list,
    max_capacity: int
) -> "tuple[list, int, list]":
    """
    Algoritm Steepest Ascent Hill-Climbing (SAHC) pentru problema rucsacului.

    Pasi:
        1. Generare solutie valida aleatoare (current hilltop c)
        2. Evaluare toti vecinii din N(c)
        3. Daca exista un vecin mai bun rezulta c = cel mai bun vecin (steepest step)
        4. Daca nu exista vecin mai bun (optim local) rezulta restart aleator (pasul 1)
        5. Returneaza cel mai bun c dupa max_evaluations evaluari

    INPUT
    ------
    max_evaluations : int
        Numarul maxim de evaluari ale functiei fitness
    objects : list[tuple[int, int]]
        Lista obiectelor ca tupluri (greutate, valoare)
    max_capacity : int
        Capacitatea maxima a rucsacului

    OUTPUT
    ------
    best_solution : list[int]
        Cea mai buna solutie gasita
    best_fitness : int
        Valoarea fitness a celei mai bune solutii
    fitness_history : list[int]
        Istoricul valorilor fitness (cea mai buna solutie la fiecare evaluare)
    """
    n = len(objects)
    evaluations_done = 0

    # Pas 1: solutie initiala aleatoare
    c = generate_valid_solution(n, objects, max_capacity)
    fc = fitness(objects, c, max_capacity)
    evaluations_done += 1

    best_solution = c[:]
    best_fitness = fc

    fitness_history = [best_fitness]

    while evaluations_done < max_evaluations:
        # Pas 2: generam TOTI vecinii lui c
        neighbors = get_all_neighbors(c)

        # Pas 3: evaluam TOTI vecinii si il gasim pe cel mai bun
        best_neighbor = None
        best_neighbor_fitness = fc  # pragul minim: trebuie sa fie mai bun decât c

        for neighbor in neighbors:
            if evaluations_done >= max_evaluations:
                break
            fn = fitness(objects, neighbor, max_capacity)
            evaluations_done += 1

            # STEEPEST: pastram cel mai bun vecin (nu primul mai bun ca in RHC)
            if fn > best_neighbor_fitness:
                best_neighbor = neighbor[:]
                best_neighbor_fitness = fn

        if best_neighbor is not None:
            # Pas 3 (continuare): exista vecin mai bun rezulta mutare la el
            c = best_neighbor[:]
            fc = best_neighbor_fitness

            # Actualizam cel mai bun global
            if fc > best_fitness:
                best_solution = c[:]
                best_fitness = fc
        else:
            # Pas 4: optim local rezulta restart aleator
            c = generate_valid_solution(n, objects, max_capacity)
            fc = fitness(objects, c, max_capacity)
            evaluations_done += 1

            if fc > best_fitness:
                best_solution = c[:]
                best_fitness = fc

        fitness_history.append(best_fitness)

    # Pas 5: returnam cel mai bun hilltop gasit
    return best_solution, best_fitness, fitness_history


# =============================================================================
# RULARE DE MULTIPLE ORI (pentru statistici)
# =============================================================================

def sahc_n_times(
    n: int,
    max_evaluations: int,
    objects: list,
    max_capacity: int
) -> "tuple[list[tuple[list, int]], float]":
    """
    Ruleaza algoritmul SAHC de n ori si colecteaza statistici.

    INPUT
    ------
    n : int
        Numarul de rulari
    max_evaluations : int
        Numarul maxim de evaluari per rulare
    objects : list[tuple[int, int]]
        Lista obiectelor
    max_capacity : int
        Capacitatea maxima a rucsacului

    OUTPUT
    ------
    best_solutions : list[tuple[list, int]]
        Lista cu (soluție, fitness) pentru fiecare rulare
    average_time : float
        Timpul mediu de executie per rulare (secunde)
    """
    best_solutions = []
    total_time = 0.0

    for _ in range(n):
        start = time.time()
        best_sol, best_fit, _ = steepest_ascent_hill_climbing(
            max_evaluations, objects, max_capacity
        )
        elapsed = time.time() - start
        total_time += elapsed
        best_solutions.append((best_sol, best_fit))

    average_time = total_time / n
    return best_solutions, average_time


def run_experiments(
    problem_instances: list,
    values_for_n: list,
    values_for_k: list
) -> str:
    """
    Ruleaza experimentele pentru toate combinatiile de parametri ai
    returneaza un tabel Markdown cu rezultatele.

    INPUT
    ------
    problem_instances : list[str]
        Lista fișierelor de date (ex: ["rucsac-20.txt", "rucsac-200.txt"])
    values_for_n : list[int]
        Valorile pentru numărul de rulări
    values_for_k : list[int]
        Valorile pentru numărul maxim de evaluări

    OUTPUT
    ------
    str
        Tabel în format Markdown cu rezultatele
    """
    markdown_table = "| Instanta | max_eval (k) | n_rulari | Medie fitness | Cel mai bun | Greutate medie | Timp mediu (s) |\n"
    markdown_table += "|----------|-------------|----------|---------------|-------------|----------------|----------------|\n"

    for pi in problem_instances:
        print(f"\nRulare pe instanța: {pi}")
        objects, max_capacity = load_data(pi)

        for n in values_for_n:
            for k in values_for_k:
                print(f"  n={n}, k={k} ... ", end="", flush=True)

                best_results, avg_time = sahc_n_times(n, k, objects, max_capacity)

                # Calculam statistici
                fitnesses = [fit for _, fit in best_results]
                avg_fitness = sum(fitnesses) / len(fitnesses)
                best_fitness = max(fitnesses)

                total_weights = []
                for sol, fit in best_results:
                    w = sum(objects[i][0] for i in range(len(sol)) if sol[i] == 1)
                    total_weights.append(w)
                avg_weight = sum(total_weights) / len(total_weights)

                print(f"best={best_fitness}, avg={avg_fitness:.1f}")

                markdown_table += (
                    f"| {pi} | {k} | {n} | {avg_fitness:.2f} | "
                    f"{best_fitness} | {avg_weight:.1f} | {avg_time:.4f} |\n"
                )

    return markdown_table


if __name__ == "__main__":
    PROBLEM_INSTANCES = ["rucsac-20.txt", "rucsac-200.txt"]

    # Numarul de rulari per combinatie de parametri
    VALUES_FOR_N = [10]

    # Numarul maxim de evaluari ale functiei fitness
    VALUES_FOR_K = [50, 100, 200, 500, 1000]

    print("\n" + "=" * 60)
    print("SAHC - Experimente complete")
    print("=" * 60)

    try:
        table = run_experiments(PROBLEM_INSTANCES, VALUES_FOR_N, VALUES_FOR_K)
        print("\nTabel rezultate (Markdown):")
        print(table)

        # Salvare tabel în fisier
        with open("rezultate_sahc.md", "w", encoding="utf-8") as f:
            f.write("# Rezultate SAHC - Problema Rucsacului\n\n")
            f.write(table)
        print("\nTabelul a fost salvat in 'rezultate_sahc.md'")

    except FileNotFoundError as e:
        print(f"Eroare la incarcarea datelor: {e}")
        print("Asigura-te ca fișierele .txt sunt prezente în directorul de lucru.")