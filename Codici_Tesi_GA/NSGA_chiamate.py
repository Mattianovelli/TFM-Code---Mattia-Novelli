import json
import itertools
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

# nohup python3 NSGA_chiamate.py > NSGA_chiamate.log 2>&1 &


# Configurazioni per tuning:
# - 1: tuning
# - 2: configurazione finale
tuning = 2
if tuning == 1:
    sigma = [0.01, 0.015, 0.02]
    eta = [0.1, 1, 10]
elif tuning == 2:
    sigma = [0.015]
    eta = [1]



# Modalità di valutazione della frequenza
# - 1: formula esatta (considerando le perdite)
# - 2: valore approssimato (senza considerare perdite)
scelta = 1
if scelta == 1:
    correction = ["no", "cor", "cor_cor"]
elif scelta == 2:
    correction = ["radice"]

# Cartella configurazioni
os.makedirs("configurazioni_NSGA", exist_ok=True)

# Percorso salvataggio csv
sub_fold = f"Tuning_{tuning}"
csv_path = os.path.join("Esperimenti_NSGA", sub_fold, f"risultati_tuning_{tuning}.csv")

# Eliminazione csv precedente
if os.path.exists(csv_path):
    os.remove(csv_path)
    print("CSV precedente eliminato")


# Esperimenti con varie configurazioni
for run_id, (eta_cross, sigma_mut, corr) in enumerate(itertools.product(eta, sigma, correction)):

    iper = {
        "run_id": run_id,
        "max_generations": 500,
        "sol_per_pop": 500,
        "scelta": scelta,
        "fase_tuning": tuning,
        "correction": corr,
        "prob_cross": 0.9,
        "eta_cross": eta_cross,
        "prob_mut": 0.25,
        "sigma_mut": sigma_mut
    }

    # Percorso configurazione
    path_config = os.path.abspath(f"configurazioni_NSGA/config_{run_id:04d}.json")

    with open(path_config, "w") as f:
        json.dump(iper, f, indent=4)

    print(rf"\nEsperimento {run_id}: $eta_c=${eta_cross}, $\sigma=${sigma_mut}")

    # Esperimento
    subprocess.run(
        ["python3", "NSGA_auto.py", path_config],
        check=True
    )
