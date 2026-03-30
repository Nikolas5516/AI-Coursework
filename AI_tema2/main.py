"""
main.py
=======
Script principal: ruleaza experimentele SA si TS pentru rucsac si TSP,
genereaza tabelele de rezultate si vizualizarile.

Structura fisiere proiect:
    main.py
    utils.py
    simulated_annealing.py
    tabu_search.py
    rucsac-20.txt
    rucsac-200.txt
    kroC100.tsp
    kroC100.opt.tour   (optional, pentru comparatie cu optim)
"""

import time
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    load_data_rucsac, read_from_file_TSP, read_from_file_TSP_opt,
    distance_matrix_TSP, plot_fitness_history, plot_tsp_route
)
from simulated_annealing import (
    simulated_annealing_rucsac, sa_rucsac_n_times,
    simulated_annealing_TSP, sa_TSP_n_times
)
from tabu_search import (
    tabu_search_rucsac, ts_rucsac_n_times,
    tabu_search_TSP, ts_TSP_n_times
)


# =============================================================================
# CONFIGURARE PARAMETRI
# =============================================================================
#
# Timp estimat rulare completa: ~3-5 minute
#
# De ce limitam parametrii:
#   - TS TSP evalueaza TOTI vecinii per iteratie:
#     2-swap/2-opt pe n=100 orase = 100*99/2 = 4950 evaluari/iteratie
#     => max_iter=100 inseamna deja 495.000 evaluari per rulare
#   - SA TSP: K_ITER iteratii per nivel temperatura, multe niveluri => creste rapid
#
# =============================================================================

N_RUNS = 5          # rulari per configuratie (pentru statistici)

# SA parametri
# K_ITER_SA = iteratii per nivel de temperatura (pastram mic)
K_ITER_SA       = 20
SA_T_VALUES     = [100, 1000, 5000]        # 3 valori temperatura initiala
SA_ALPHA_VALUES = [0.99, 0.95, 0.80]       # 3 valori coeficient racire
SA_TMIN_VALUES  = [0.1, 0.01]              # 2 valori temperatura minima
# Total SA rucsac: 2 fisiere x 3 x 3 x 2 x 5 rulari = 180 rulari (rapide)
# Total SA TSP:    2 metode  x 3 x 3 x 2 x 5 rulari = 180 rulari (~1-2 min)

# TS parametri
# IMPORTANT: TS TSP e cel mai lent - limitam max_iter la valori mici
TS_MAX_ITER_VALUES_RUCSAC = [50, 100, 200, 500, 1000]   # rucsac e rapid
TS_MAX_ITER_VALUES_TSP    = [10, 20, 50, 100, 200]       # TSP e lent (4950 eval/iter)
TS_TENURE_VALUES          = [3, 5, 10]                   # 3 valori tenure
# Total TS rucsac: 2 x 5 x 3 x 5 = 150 rulari (rapide)
# Total TS TSP:    2 x 5 x 3 x 5 = 150 rulari, dar max_iter mic => ~2-3 min

RUCSAC_FILES = ["rucsac-20.txt", "rucsac-200.txt"]
TSP_FILE     = "kroC100.tsp"
TSP_OPT_FILE = "kroC100.opt.tour"


# =============================================================================
# HELPERS
# =============================================================================

def print_table_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def stats(results):
    """Calculeaza best, medie si worst dintr-o lista de (sol, fit)."""
    fits = [f for _, f in results]
    return max(fits), sum(fits) / len(fits), min(fits)


# =============================================================================
# EXPERIMENTE SA - RUCSAC
# =============================================================================

def run_sa_rucsac():
    print_table_header("SA - RUCSAC")
    header = f"{'Instanta':<20} {'T':>6} {'alpha':>6} {'t_min':>8} {'K':>5} | {'Best':>8} {'Avg':>10} {'Worst':>8} {'Timp(s)':>9}"
    print(header)
    print("-" * len(header))

    for fname in RUCSAC_FILES:
        try:
            objects, max_capacity = load_data_rucsac(fname)
        except FileNotFoundError:
            print(f"  [SKIP] {fname} nu a fost gasit.")
            continue

        for T in SA_T_VALUES:
            for alpha in SA_ALPHA_VALUES:
                for t_min in SA_TMIN_VALUES:
                    results, avg_time = sa_rucsac_n_times(
                        N_RUNS, T, alpha, t_min, K_ITER_SA, objects, max_capacity
                    )
                    best, avg, worst = stats(results)
                    print(f"{fname:<20} {T:>6} {alpha:>6.2f} {t_min:>8.5f} {K_ITER_SA:>5} | {best:>8} {avg:>10.1f} {worst:>8} {avg_time:>9.4f}")

    print()


# =============================================================================
# EXPERIMENTE SA - TSP
# =============================================================================

def run_sa_TSP():
    print_table_header("SA - TSP (kroC100)")
    try:
        locations, dimension = read_from_file_TSP(TSP_FILE)
        dm = distance_matrix_TSP(locations)
    except FileNotFoundError:
        print(f"  [SKIP] {TSP_FILE} nu a fost gasit.")
        return

    # Optim cunoscut (daca exista fisierul)
    try:
        opt_sol = read_from_file_TSP_opt(TSP_OPT_FILE)
        from utils import fitness_TSP
        opt_dist = fitness_TSP(opt_sol, dm)
        print(f"  Distanta optima cunoscuta: {opt_dist}")
    except FileNotFoundError:
        opt_dist = None

    for use_opt in [False, True]:
        method = "2-opt" if use_opt else "2-swap"
        print(f"\n  Vecinatate: {method}")
        header = f"  {'T':>6} {'alpha':>6} {'t_min':>8} {'K':>5} | {'Best':>8} {'Avg':>10} {'Gap%':>7} {'Timp(s)':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for T in SA_T_VALUES:
            for alpha in SA_ALPHA_VALUES:
                for t_min in SA_TMIN_VALUES:
                    results, avg_time = sa_TSP_n_times(
                        N_RUNS, T, alpha, t_min, K_ITER_SA, dm, dimension, use_opt
                    )
                    best, avg, worst = stats(results)
                    best = min(f for _, f in results)  # TSP: minimizare
                    avg  = sum(f for _, f in results) / len(results)
                    gap  = f"{(best/opt_dist - 1)*100:.1f}%" if opt_dist else "N/A"
                    print(f"  {T:>6} {alpha:>6.2f} {t_min:>8.5f} {K_ITER_SA:>5} | {best:>8} {avg:>10.0f} {gap:>7} {avg_time:>9.2f}")

    print()


# =============================================================================
# EXPERIMENTE TS - RUCSAC
# =============================================================================

def run_ts_rucsac():
    print_table_header("TS - RUCSAC")
    header = f"{'Instanta':<20} {'max_iter':>8} {'tenure':>7} | {'Best':>8} {'Avg':>10} {'Worst':>8} {'Timp(s)':>9}"
    print(header)
    print("-" * len(header))

    for fname in RUCSAC_FILES:
        try:
            objects, max_capacity = load_data_rucsac(fname)
        except FileNotFoundError:
            print(f"  [SKIP] {fname} nu a fost gasit.")
            continue

        for max_iter in TS_MAX_ITER_VALUES_RUCSAC:
            for tenure in TS_TENURE_VALUES:
                results, avg_time = ts_rucsac_n_times(
                    N_RUNS, max_iter, tenure, objects, max_capacity
                )
                best, avg, worst = stats(results)
                print(f"{fname:<20} {max_iter:>8} {tenure:>7} | {best:>8} {avg:>10.1f} {worst:>8} {avg_time:>9.4f}")

    print()


# =============================================================================
# EXPERIMENTE TS - TSP
# =============================================================================

def run_ts_TSP():
    print_table_header("TS - TSP (kroC100)")
    try:
        locations, dimension = read_from_file_TSP(TSP_FILE)
        dm = distance_matrix_TSP(locations)
    except FileNotFoundError:
        print(f"  [SKIP] {TSP_FILE} nu a fost gasit.")
        return

    try:
        opt_sol = read_from_file_TSP_opt(TSP_OPT_FILE)
        from utils import fitness_TSP
        opt_dist = fitness_TSP(opt_sol, dm)
        print(f"  Distanta optima cunoscuta: {opt_dist}")
    except FileNotFoundError:
        opt_dist = None

    for use_opt in [False, True]:
        method = "2-opt" if use_opt else "2-swap"
        print(f"\n  Vecinatate: {method}")
        header = f"  {'max_iter':>8} {'tenure':>7} | {'Best':>8} {'Avg':>10} {'Gap%':>7} {'Timp(s)':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for max_iter in TS_MAX_ITER_VALUES_TSP:
            for tenure in TS_TENURE_VALUES:
                results, avg_time = ts_TSP_n_times(
                    N_RUNS, max_iter, tenure, dm, dimension, use_opt
                )
                best = min(f for _, f in results)
                avg  = sum(f for _, f in results) / len(results)
                gap  = f"{(best/opt_dist - 1)*100:.1f}%" if opt_dist else "N/A"
                print(f"  {max_iter:>8} {tenure:>7} | {best:>8} {avg:>10.0f} {gap:>7} {avg_time:>9.2f}")

    print()


# =============================================================================
# VIZUALIZARE - evolutie fitness intr-o singura rulare
# =============================================================================

def run_visualizations():
    print_table_header("VIZUALIZARE - Evolutie fitness")

    # --- Rucsac: SA vs TS ---
    try:
        objects, max_capacity = load_data_rucsac("rucsac-200.txt")
        n = len(objects)

        _, _, sa_hist = simulated_annealing_rucsac(1000, 0.99, 0.01, K_ITER_SA, objects, max_capacity)
        _, _, ts_hist = tabu_search_rucsac(500, 5, objects, max_capacity)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(sa_hist, color="steelblue", linewidth=0.8)
        axes[0].set_title("SA - rucsac-200.txt")
        axes[0].set_xlabel("Iteratie")
        axes[0].set_ylabel("Fitness (valoare)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts_hist, color="darkorange", linewidth=0.8)
        axes[1].set_title("TS - rucsac-200.txt")
        axes[1].set_xlabel("Iteratie")
        axes[1].set_ylabel("Fitness (valoare)")
        axes[1].grid(True, alpha=0.3)

        plt.suptitle("Evolutie fitness - Problema Rucsacului", fontsize=13)
        plt.tight_layout()
        plt.savefig("evolutie_rucsac.png", dpi=120)
        plt.show()
        print("  Grafic salvat: evolutie_rucsac.png")

    except FileNotFoundError:
        print("  [SKIP] rucsac-200.txt nu a fost gasit.")

    # --- TSP: SA 2-swap vs SA 2-opt vs TS 2-swap vs TS 2-opt ---
    try:
        locations, dimension = read_from_file_TSP(TSP_FILE)
        dm = distance_matrix_TSP(locations)

        _, _, sa_swap_hist = simulated_annealing_TSP(1000, 0.99, 0.01, K_ITER_SA, dm, dimension, use_two_opt=False)
        _, _, sa_opt_hist  = simulated_annealing_TSP(1000, 0.99, 0.01, K_ITER_SA, dm, dimension, use_two_opt=True)
        _, _, ts_swap_hist = tabu_search_TSP(200, 5, dm, dimension, use_two_opt=False)
        _, _, ts_opt_hist  = tabu_search_TSP(200, 5, dm, dimension, use_two_opt=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        configs = [
            (sa_swap_hist, "SA 2-swap", "steelblue"),
            (sa_opt_hist,  "SA 2-opt",  "royalblue"),
            (ts_swap_hist, "TS 2-swap", "darkorange"),
            (ts_opt_hist,  "TS 2-opt",  "tomato"),
        ]
        for ax, (hist, label, color) in zip(axes.flat, configs):
            ax.plot(hist, color=color, linewidth=0.8)
            ax.set_title(label)
            ax.set_xlabel("Iteratie")
            ax.set_ylabel("Distanta")
            ax.grid(True, alpha=0.3)

        plt.suptitle("Evolutie distanta TSP - kroC100", fontsize=13)
        plt.tight_layout()
        plt.savefig("evolutie_tsp.png", dpi=120)
        plt.show()
        print("  Grafic salvat: evolutie_tsp.png")

        # Ruta TSP cu TS 2-opt
        best_sol_ts_opt, _, _ = tabu_search_TSP(300, 5, dm, dimension, use_two_opt=True)
        plot_tsp_route(locations, best_sol_ts_opt, title="Ruta TSP - TS 2-opt (kroC100)")

    except FileNotFoundError:
        print(f"  [SKIP] {TSP_FILE} nu a fost gasit.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  TEMA 2 - Simulated Annealing si Tabu Search")
    print("  Rucsac (20 + 200 obiecte) + TSP (kroC100)")
    print("=" * 70)

    # Comenteaza/decomenteza ce vrei sa rulezi:
    run_sa_rucsac()
    run_ts_rucsac()
    run_sa_TSP()
    run_ts_TSP()
    run_visualizations()