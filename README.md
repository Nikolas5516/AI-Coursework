# Artificial Intelligence Coursework 🤖

This repository contains my laboratory assignments for the Artificial Intelligence course (Spring 2026).
It focuses on implementing and comparing various search algorithms and metaheuristics for optimization problems.

## 📁 Repository Structure

* **[AI_tema1](./AI_tema1): Local Search - Knapsack Problem**
  * Implementation of **Steepest Ascent Hill-Climbing (SAHC)** for the Knapsack problem.
  * Comparison with Random Hill-Climbing (RHC).
  * Analysis across multiple instances with different parameter configurations.

* **[AI_tema2](./AI_tema2): Metaheuristics - TSP & Knapsack**
  * Implementation of **Tabu Search (TS)** and **Simulated Annealing (SA)**.
  * Application on the Traveling Salesman Problem (TSP) using **2-swap** and **2-opt** movement operators.
  * Comparative analysis between HC, SA, and TS for the Knapsack problem.
  * Performance visualization across iterations and parameter tuning.

* **[AI_tema3](./AI_tema3): Evolutionary Algorithms & Swarm Intelligence**
  * Implementation of an **Evolutionary Algorithm (EA) for real-valued encoding** applied to the Griewank function (problem 8).
  * Implementation of **Particle Swarm Optimization (PSO)** for the same real-valued problem.
  * EA applied to the **Knapsack problem** (binary encoding with solution repair) and **TSP** (permutation encoding with Order Crossover).
  * Comparative analysis across all algorithms from previous assignments (RHC, SAHC, SA, TS, EA, PSO).
  * Developed as a **Jupyter Notebook** (`.ipynb`) for interactive execution and inline visualization.

## 🛠️ Technologies & Tools

* **Language:** Python 3.x
* **Environment:** PyCharm (scripts for tema 1 & 2, Jupyter Notebook for tema 3)
* **Libraries:**
  * `NumPy`: For numerical operations and data handling.
  * `Matplotlib`: For visualizing algorithm evolution and performance graphs.
  * `nbformat`: For Jupyter Notebook support.
* **Problem Instances:** Standard Knapsack and TSP datasets (e.g., `kroC100.tsp`, `rucsac-20.txt`, `rucsac-200.txt`).