import json
import itertools
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# nohup python3 GA_chiamate.py > GA_chiamate.log 2>&1 &


# Configurazioni per tuning:
# - 1: prima fase di tuning
# - 2: seconda fase di tuning
# - 3: configurazione finale
tuning = 1
if tuning == 1:
    p_M_min = [0.15, 0.2, 0.25]
    genitori = [0.2, 0.4, 0.6]
    k_tornei = [2, 3, 4]
elif tuning == 2:
    p_M_min = [0.225, 0.25, 0.275]
    genitori = [0.6]
    k_tornei = [3]
elif tuning == 3:
    p_M_min = [0.275]
    genitori = [0.6]
    k_tornei = [3]

# Modalità di valutazione della frequenza
# - 1: formula esatta (considerando le perdite)
# - 2: valore approssimato (senza considerare perdite)
scelta = 2
if scelta == 1:
    correction = ["no", "cor", "cor_cor"]
elif scelta == 2:
    correction = ["radice"]


# Cartella configurazioni
os.makedirs("configurazioni_GA", exist_ok=True)

# Percorso salvataggio csv
sub_fold = f"Tuning_{tuning}"
csv_path = os.path.join("Esperimenti", sub_fold, f"risultati_tuning_{tuning}.csv")

# Eliminazione csv precedente
if os.path.exists(csv_path):
    os.remove(csv_path)
    print("CSV precedente eliminato")


# Esperimenti
for run_id, (p_M_min_val, parents, k, corr) in enumerate(itertools.product(p_M_min, genitori, k_tornei, correction)):

    iper = {
        "run_id": run_id,
        "num_generations": 25,
        "sol_per_pop": 500,
        "parent_ratio": parents,
        "K_tournament": k,
        "elitism_ratio": 0.05,
        "keep_parent_ratio": 0.05,
        "p_M_max": 0.5,
        "p_M_min": p_M_min_val,
        "tipo_fitness": 4,
        "scelta": scelta,
        "fase_tuning": tuning,
        "correction": corr
    }

    # Percorso configurazioni
    path_config = os.path.abspath(f"configurazioni/config_{run_id:04d}.json")

    with open(path_config, "w") as f:
        json.dump(iper, f, indent=4)

    print(rf"\nEsperimento {run_id}: $p_{M,\min}=${p_M_min_val}, parents={parents}, $k=${k}")

    # Esperimento
    subprocess.run(
        ["python3", "GA_auto.py", path_config],
        check=True
    )
