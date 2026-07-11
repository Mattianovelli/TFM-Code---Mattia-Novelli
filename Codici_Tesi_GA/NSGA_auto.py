import pygad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from Codici_ausiliari.air_features import air_features
from Codici_ausiliari.resolution_losses import resolution_losses
from dataclasses import dataclass
from paretoset import paretoset

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2, DefaultMultiObjectiveTermination
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SBX
from pymoode.performance import SpacingIndicator as SP

import json
import sys
import csv

# nohup python3 NSGA_auto.py > NSGA_auto.log 2>&1 &

######################################################################################
# Importo gli iperparametri
config_file = sys.argv[1] 
with open(config_file) as f:
    iper = json.load(f)


# Plot ad ogni generazione:
# - 1: si
# - altro: no
disegno = 0


# Lettura dati
scelta = iper["scelta"]
tipo_fitness = iper["tipo_fitness"]
tuning = iper["fase_tuning"]
correction = iper.get("correction") 
run_id = iper.get("run_id", 0)

#################################################################################
# Percorsi per salvare plot e csv
cartella_base = "Esperimenti_NSGA"
cartella_generale = os.path.join(cartella_base, f"Tuning_{tuning}") 
os.makedirs(cartella_generale, exist_ok=True)

# Differenzio a seconda della modalità di calcolo della frequenza
if scelta == 1:
    cartella = os.path.join(cartella_generale, f"Risultati_{run_id:04d}_{correction}")
else:
    cartella = os.path.join(cartella_generale, f"Risultati_approx_{run_id:04d}_{correction}")

# Sotto cartella grafici ad ogni iterazione
os.makedirs(cartella, exist_ok=True)
cartella_plot = os.path.join(cartella, "Plot")
os.makedirs(cartella_plot, exist_ok=True)

#####################################################################################
# IPERPARAMETRI E COSTANTI
# Costanti dell'aria
air = air_features()
c0 = air.c0

# Parametri per metodo delle secanti
tol = 1e-8
maxit = 100
x0 = 100
x1 = 2 * x0

# Forma del risonatore (3: cilindrico)
geom = 3
  
#####################################################################################
# Strutture per neck e cavity
@dataclass
class Neck:
    l: float
    dl: float
    r: float

@dataclass
class Cavity:
    l: float
    dl: float
    r: float

########################################################################################
# Dominio Omega: ln, rn, lc, rc
bound_min = np.array([0.01, 0.001, 0.01, 0.001])
bound_max = np.array([0.08, 0.04, 0.08, 0.04])

########################################################################################
# Funzioni obiettivo:
# - f1: volume
# - f2: frequenza angolare di risonanza

def valutazione_funzioni(solution):

    # Parametri risonatore
    l_n, r_n, l_c, r_c = solution

    # f1 = volume
    f1 = np.pi * (r_n**2 * l_n + r_c**2 * l_c)

    # f2 = frequenza di risonanza
    if r_n >= 0.55 * r_c:
            f2 = 1e10
            f1 = 1e10
    else:
        if scelta == 1:

            # End-corrections
            rr = r_n / r_c
            if correction == "no":
                dl_c = 0.0
                dl_n = 0.0
            elif correction == "cor":
                dl_c = 0.82 * (1 - 1.35 * rr + 0.31 * rr**3) * r_n
                dl_n = 0
            elif correction == "cor_cor":
                dl_c = 0.82 * (1 - 1.35 * rr + 0.31 * rr**3) * r_n
                dl_n = 0.82 * r_n

            # Collo e cavità
            neck = Neck(l=l_n, dl=dl_n, r=r_n)
            cavity = Cavity(l=l_c, dl=dl_c, r=r_c)

            # Calcolo frequenza con il metodo delle secanti
            f2 = resolution_losses(neck, cavity, air, geom, x0, x1, tol, maxit)
            
        elif scelta == 2:
            f2 = c0 * np.sqrt((r_n**2) / (l_n * l_c * r_c**2))
    
    return f1, f2


#######################################################################################
# Importo il csv del pareto uscito dalla tabulazione
if scelta == 1:
    df = pd.read_csv("tabulazione_pareto.csv")
else:
    df = pd.read_csv("tabulazione_pareto_approx.csv")

pareto_vero = df[["f1_volume", "f2_freq"]].to_numpy()

######################################################################################
# Indicatore di spacing
spacing = SP()

######################################################################################
# Problema
class problema_risonatore(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var = 4,
            n_obj = 2,
            n_ieq_constr = 0,
            xl = bound_min,
            xu = bound_max
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2 = valutazione_funzioni(x)
        out["F"] = [f1, f2]


#######################################################################################################
# IPERPARAMETRI
prob_cross = iper["prob_cross"]
eta_cross = iper["eta_cross"]
prob_mut = iper["prob_mut"]
sigma_mut = iper["sigma_mut"]

# OPERATORI
crossover = SBX(prob = prob_cross, eta = eta_cross)
mutation = GaussianMutation(sigma = sigma_mut, prob = prob_mut)
#mutation = PolynomialMutation(prob = 0.2, eta = 20)

# ALGORITMO
algoritmo = NSGA2(pop_size = 500, 
                  crossover = crossover, 
                  mutation = mutation
                  )

# CRITERIO D'ARRESTO: numero generazioni + avanzamento stagnante dei valori di f
arresto = DefaultMultiObjectiveTermination(
    xtol=1e-6,
    #cvtol=1e-6,
    ftol=0.005,
    period=10,
    n_max_gen=500
)

# ACQUISIZIONE PROBLEMA
problema = problema_risonatore()

# CICLO PER STABILITà
risultati_n_generazioni = []
risultati_spacing = []

if tuning == 2:
    n_rep = 1
elif tuning ==1:
    n_rep = 10

#N_RIPETIZIONI = 1

for ripetizione in range(n_rep):
    print(f"\n\nRIPETIZIONE {ripetizione + 1}/{n_rep}")

    # RISOLUZIONE
    res = minimize(problema, algoritmo, arresto, save_history = True, verbose = True)


    ########################################################################################
    # PLOT PER OGNI GENERAZIONE
    print("\n Grafici per generazione")

    # dati
    history = res.history


    if ((disegno == 1) & (n_rep == 1)):
        for i, algorithm_gen in enumerate(tqdm(history)):
            gen_num = i + 1
            
            # Estrazione popolazione
            pop_gen = algorithm_gen.pop
            F_gen = pop_gen.get("F")
            vol_gen = F_gen[:, 0]
            omega_gen = F_gen[:, 1]
            mask_gen = (vol_gen < 1e9) & (omega_gen < 1e9)
            
            # Estrazione fronte
            opt_gen = algorithm_gen.opt
            F_opt_gen = opt_gen.get("F")
            vol_opt_gen = F_opt_gen[:, 0]
            omega_opt_gen = F_opt_gen[:, 1]
            mask_opt_gen = (vol_opt_gen < 1e9) & (omega_opt_gen < 1e9)
            
            # Plot
            plt.figure(figsize=(8, 6))
            plt.scatter(vol_gen[mask_gen], omega_gen[mask_gen], color="orange", s=10, alpha=0.6, label="Dominated Solutions")
            plt.scatter(vol_opt_gen[mask_opt_gen], omega_opt_gen[mask_opt_gen], color="red", s=15, label="Pareto Front (Non-Dominated)")
            
            plt.xlabel(rf"$V$", fontsize = 14)
            plt.ylabel(rf"$\omega$", fontsize = 14)
            plt.title(f"NSGA-II - Generation {gen_num}", fontsize = 16)
            plt.legend(fontsize = 12)
            plt.grid(True, linestyle="--", alpha=0.5)
            
            # Salvo
            nome_plot_gen = f"plot_gen_{gen_num:03d}.png"
            percorso_plot_gen = os.path.join(cartella_plot, nome_plot_gen)
            plt.savefig(percorso_plot_gen, dpi=150, bbox_inches = 'tight')
            plt.close()


    ########################################################################################################
    # RISULTATI
    # Variabili
    X = res.X       

    # Funzioni obiettivo
    F = res.F
    volume = F[:,0]
    omega = F[:,1]

    # Funzione obiettivo totale
    algoritmo_finito = res.algorithm
    F_pop = algoritmo_finito.pop.get("F")
    volume_pop = F_pop[:, 0]
    omega_pop = F_pop[:, 1]


    ########################################################################################################
    # CALCOLO STATISCO
    # Normalizzo
    F_min = F.min(axis=0)
    F_max = F.max(axis=0)
    F_normalizzato = (F - F_min) / (F_max - F_min)

    # Calcolo spacing sul fronte scalato
    spacing_finale = spacing.do(F_normalizzato)

    # Altre Statistiche
    n_generazioni = res.algorithm.n_gen
    n_non_dominate = len(volume)

    # Aggiungo i valori nel vettore
    risultati_n_generazioni.append(n_generazioni)
    risultati_spacing.append(spacing_finale)

    ########################################################################################################
    # PLOT
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(pareto_vero[:, 0], pareto_vero[:, 1], alpha = 0.5, s = 8, label = "True Front")
    ax.scatter(volume_pop, omega_pop, color = "orange", s = 10, label = "Dominated Solutions")
    ax.scatter(volume, omega, color = "red", s = 10, label = "Non-Dominated Solutions")

    ax.set_xlabel(r"Volume ($V$ [m$^3$])", fontsize = 14)
    ax.set_ylabel(r"Angular Frequency ($\omega$ [rad/s])", fontsize = 14)
    ax.set_title("True Pareto Front vs NSGA-II Approximation", fontsize = 16)
    ax.grid(True, linestyle = "--", alpha = 0.5)

    # Box
    testo_box = (
        f"Generations: {n_generazioni},    Non-Dominated Solutions: {n_non_dominate}/{len(volume_pop)}"
    )
    stile_box = dict(boxstyle = 'round,pad=0.6', facecolor = 'white', edgecolor = 'gray', alpha = 1)
    ax.text(0.5, -0.16, testo_box, ha = 'center', transform = ax.transAxes, fontsize = 12,
            verticalalignment = 'top', bbox = stile_box, linespacing = 1.4)

    # Legenda
    ax.legend(loc="upper right", fontsize=12)

    # Salvo
    confronto_fronte = os.path.join(cartella, rf"Confronto_Fronte_NSGA_{ripetizione}_{correction}.png")
    plt.savefig(confronto_fronte, dpi = 300, bbox_inches = 'tight')
    print(f"Grafico salvato in: {confronto_fronte}")
    plt.close()



#########################################################################################################
# CSV
# Salvo solo se faccio un unico esperimento
if n_rep == 1:
    df = pd.DataFrame({

        "l_n": X[:,0],
        "r_n": X[:,1],
        "l_c": X[:,2],
        "r_c": X[:,3],

        "f1_volume": F[:,0],
        "f2_freq": F[:,1]
    })

    # Riordino
    df = df.sort_values(by = "f1_volume", ascending = True).reset_index(drop = True)
    csv_pareto = os.path.join(cartella, rf"pareto_nsga2_{correction}.csv")
    df.to_csv(csv_pareto, index=False)

########################################################################################################
# Calcolo medie
media_n_generazioni = round(np.mean(risultati_n_generazioni))
media_spacing = np.mean(risultati_spacing)


# Stampe
print(f"\n\nVALORI MEDI METRICHE")
print(f"Numero medio di generazioni: \t\t {media_n_generazioni}")
print(f"Media spacing: {media_spacing}")

###########################################################################################################################
# Salvo CSV per confronti nel tuning
csv_path = os.path.join(cartella_generale, f"risultati_tuning_{tuning}.csv")

def salva_csv_tuning(run_id, eta_c, sigma, n_generazioni, spacing):

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # header solo la prima volta
        if not file_exists:
            writer.writerow(["run_id", "eta_c", "sigma", "n_generazioni", "spacing"])

        writer.writerow([run_id, eta_c, sigma, n_generazioni, spacing])

# Salvo CSV
salva_csv_tuning(
    run_id = run_id, 
    eta_c = iper["eta_cross"], 
    sigma = iper["sigma_mut"], 
    n_generazioni = media_n_generazioni, 
    spacing = media_spacing, 
)
