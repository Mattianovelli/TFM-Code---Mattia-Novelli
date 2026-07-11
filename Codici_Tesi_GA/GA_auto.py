import pygad
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from Codici_ausiliari.air_features import air_features
from Codici_ausiliari.resolution_losses import resolution_losses
from dataclasses import dataclass
from GA_funzioni.blend_crossover import blend_crossover
from GA_funzioni.gaussian_mutation_mia import gaussian_mutation_mia
from paretoset import paretoset
import json
import sys
import csv
from pymoo.indicators.hv import Hypervolume

# nohup python3 GA_auto.py > GA_auto.log 2>&1 &

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

###################################################################################
# Percorsi per salvare plot e csv
cartella_base = "Esperimenti_GA"
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
# Dominio Omega
Omega = [
    {"low": 0.01,  "high": 0.08},  # l_n
    {"low": 0.001, "high": 0.04},  # r_n
    {"low": 0.01,  "high": 0.08},  # l_c
    {"low": 0.001, "high": 0.04},  # r_c
]

# Volume massimo
V_max = np.pi * (
    Omega[1]["high"]**2 * Omega[0]["high"] +
    Omega[3]["high"]**2 * Omega[2]["high"]
)


# Stima della frequenza massima
omega_max = c0 * np.sqrt((Omega[1]["high"]**2) / (Omega[0]["low"] * Omega[2]["low"] * Omega[3]["low"]**2 ))

# Coefficienti per la fitness function
a = 1 / V_max
b = 450 / omega_max

#########################################################################################
# COFRONTO CON IL VERO FRONTE
# Importo il csv del pareto uscito dalla tabulazione
if scelta == 1:
    df = pd.read_csv("tabulazione_pareto.csv")
else:
    df = pd.read_csv("tabulazione_pareto_approx.csv")

pareto_vero = df[["f1_volume", "f2_freq"]].to_numpy()

# Filtro nella zona d'interesse
maschera_vera = (pareto_vero[:, 0] < 0.0002) & (pareto_vero[:, 1] < 1500)
maschera_vera2 = (pareto_vero[:, 0] < 0.0001) & (pareto_vero[:, 1] < 500)

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
            f2 = +np.inf
            f1 = +np.inf
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


######################################################################################
# FITNESS FUNCTION
tipo_fitness = iper["tipo_fitness"]
def fitness_function(_, punti,__):

    f1, f2 = valutazione_funzioni(punti)
    if tipo_fitness == 1:
        fitness = 1 / (a * f1 + b * f2)
    elif tipo_fitness == 2:
        fitness = 1 / np.sqrt(a * f1 + b * f2)
    elif tipo_fitness == 3:
        fitness = -np.sqrt(1 + a * f1 + b * f2)
    elif tipo_fitness == 4:
        fitness = -(a * f1 + b * f2)
    return fitness


########################################################################################
# Ipervolume e numero di punti nella zona di interesse

def calcola_ipervolume_filtrato(F, V_tilde = 1e-4, omega_tilde = 500):
    # Filtraggio punti nella zona d'intersse
    maschera = (F[:, 0] <= V_tilde) & (F[:, 1] <= omega_tilde)
    punti_filtrati = F[maschera]

    # Punto di massimo della fitness (per spezzare gli ipervolumi)
    V_cut = 4.34 * 1e-5
    omega_cut = 326
    
    # Conteggio punti filtrati
    n_punti = len(punti_filtrati)
    
    # Caso 1
    punto_ref_1   = np.array([V_tilde, omega_cut])
    opt_1         = Hypervolume(ref_point = punto_ref_1)
    hypervolume_1 = opt_1.do(punti_filtrati)

    # Caso 2
    punto_ref_2   = np.array([V_cut, omega_tilde])
    opt_2         = Hypervolume(ref_point = punto_ref_2)
    hypervolume_2 = opt_2.do(punti_filtrati)
    
    # Metto assieme
    hypervolume = hypervolume_1 + hypervolume_2
            
    return hypervolume, n_punti

########################################################################################
# Barra con tqdm ed eventuali plot ad ogni generazione

pbar = None

# Probabilità di mutazione iniziale e finale
p_M_max = iper["p_M_max"]
p_M_min = iper["p_M_min"]

# on generation
def on_generation(situazione_ga):
    global pbar

    # Stampo la miglior soluzione attuale
    solution, fitness, _ = situazione_ga.best_solution()
    f1, f2 = valutazione_funzioni(solution)

    pbar.set_postfix(miglior_fitness = f"{fitness:.3e}", V = f"{f1:.2e}", omega = f"{f2:.2f}")
    pbar.update(1)

    # Controllo della probabilità di mutazione
    n_gen_tot = situazione_ga.num_generations
    n_gen_attuale = situazione_ga.generations_completed
    p_M_i = p_M_max - (p_M_max - p_M_min) * ((n_gen_attuale) / (n_gen_tot-1))
    situazione_ga.mutation_probability = p_M_i


    # Eventuali plot intermedi
    if disegno == 1 and (situazione_ga.generations_completed % 5 == 0 or situazione_ga.generations_completed == 1):

        population = situazione_ga.population

        f1_eval = []
        f2_eval = []

        for i in population:
            f1_i, f2_i = valutazione_funzioni(i)
            f1_eval.append(f1_i)
            f2_eval.append(f2_i)

        plt.figure()

        plt.scatter(f1_eval, f2_eval, alpha=0.5)

        plt.xlabel(rf"$V$", fontsize = 14)
        plt.ylabel(rf"$\omega$", fontsize = 14)
        plt.title(f"Generation {situazione_ga.generations_completed}", fontsize = 20)

        # Salvataggio
        titolo = os.path.join(cartella_plot, f"gen_{situazione_ga.generations_completed:04d}.png")
        plt.savefig(titolo, dpi=300)

        plt.close()



#######################################################################################################
# ALGORITMO GENETICO

# Iperparametri GA
num_generations = iper["num_generations"]
sol_per_pop = iper["sol_per_pop"]
num_parents_mating = round(iper["parent_ratio"] * sol_per_pop)
parent_selection_type = "tournament"
K_tournament = iper["K_tournament"]
keep_elitism = round(iper["elitism_ratio"] * sol_per_pop)
keep_parents = round(iper["keep_parent_ratio"] * num_parents_mating)
tipo_fitness = iper["tipo_fitness"]
scelta = iper["scelta"]
num_genes = 4
gene_space = Omega
mutation_type = gaussian_mutation_mia
crossover_type = blend_crossover

# Vettori in cui salvo i risultati ad ogni iterazione (per poter calcolare i valori medi delle metriche usate nel tuning)
risultati_n_pareto = []
risultati_n_pareto_filtrato = []
risultati_hypervolume = []

# Scelta del numero di esperimenti con una stessa configurazione:
# - 15 esperimenti nelle fasi di tuning
# - 1 esperimento con la configurazione finale
if tuning == 3:
    n_rep = 1
else:
    n_rep = 15

# Risoluzione
for ripetizione in range(n_rep):
    print(f"\n\nRIPETIZIONE {ripetizione + 1}/{n_rep}")

    # Algoritmo genetico
    ga = pygad.GA(
        num_generations       = num_generations,
        num_parents_mating    = num_parents_mating,
        parent_selection_type = parent_selection_type,
        K_tournament          = K_tournament,
        fitness_func          = fitness_function,
        sol_per_pop           = sol_per_pop,
        num_genes             = num_genes,
        gene_space            = gene_space, 
        mutation_type         = mutation_type,
        mutation_probability  = p_M_max,
        keep_elitism          = keep_elitism,         
        keep_parents          = keep_parents,                 
        crossover_type        = crossover_type,
        on_generation         = on_generation
        )


    ########################################################################################################
    # Main
    pbar = tqdm(total=ga.num_generations, desc="GA Progress")

    ga.run()

    pbar.close()

    miglior_soluzione, fitness, indici = ga.best_solution()

    print("\nMiglior soluzione:")
    print(miglior_soluzione)
    [f1_best, f2_best] = valutazione_funzioni(miglior_soluzione)
    print("\nValori obiettivo:")
    print(f1_best, f2_best)
    print("fitness:", fitness)


    ####################################################################################
    # Estrazione popolazione finale
    pop_fin = ga.population

    f1_eval = []
    f2_eval = []

    for i in pop_fin:
        f1, f2 = valutazione_funzioni(i)
        f1_eval.append(f1)
        f2_eval.append(f2)

    ga_f = np.array(list(zip(f1_eval, f2_eval)))

    # Pareto
    pareto_maschera = paretoset(ga_f, sense=["min", "min"])
    pareto_front = ga_f[pareto_maschera]

    # Metriche: numero di punti non dominati, numero di punti filtrati e ipervolume
    n_pareto = int(np.sum(pareto_maschera))
    hypervolume, n_pareto_filtrato = calcola_ipervolume_filtrato(pareto_front, ref_V=1e-4, ref_omega=500.0)

    # Salvataggio nei vettori
    risultati_n_pareto.append(n_pareto)
    risultati_n_pareto_filtrato.append(n_pareto_filtrato)
    risultati_hypervolume.append(hypervolume)

    #######################################################################################################

    # Plot risultati 
    plt.figure()

    plt.scatter(f1_eval, f2_eval, label="Dominated solutions", s = 5)

    plt.scatter(ga_f[pareto_maschera, 0], ga_f[pareto_maschera, 1], color="red",
                label="Non-dominated solutions", s = 5)

    plt.xlabel(r"Volume $V$", fontsize = 14)
    plt.ylabel(r"Angular Frequency $\omega$", fontsize = 14)
    plt.legend(fontsize = 12)

    grafico_completo = os.path.join(cartella, f"GA_risultati_{ripetizione}_{correction}.png")
    plt.savefig(grafico_completo, dpi=300)
    plt.show()
    

    
##########################################################
# SALVATAGGIO SOLUZIONI DEL FRONTE E NON (solo nella configurazione definitiva)
if tuning == 3:
    
    # PUNTI DOMINATI
    # Non pareto
    non_pareto_maschera = ~pareto_maschera

    non_pareto_individui = pop_fin[non_pareto_maschera]
    non_pareto_funzioni_obiettivo = ga_f_obj[non_pareto_maschera]

    # Creo DataFrame
    df_non_pareto = pd.DataFrame({
        "l_n": non_pareto_individui[:, 0],
        "r_n": non_pareto_individui[:, 1],
        "l_c": non_pareto_individui[:, 2],
        "r_c": non_pareto_individui[:, 3],
        "f1_volume": non_pareto_funzioni_obiettivo[:, 0],
        "f2_freq": non_pareto_funzioni_obiettivo[:, 1],
    })

    # Salvataggio CSV
    df_non_pareto.to_csv(os.path.join(cartella, f"non_pareto_solutions.csv"), index=False)

    # PUNTI NON DOMINATI
    # Pareto
    pareto_individui = pop_fin[pareto_maschera]
    pareto_funzioni_obiettivo = ga_f_obj[pareto_maschera]

    # Creo DataFrame
    df_pareto = pd.DataFrame({
        "l_n": pareto_individui[:, 0],
        "r_n": pareto_individui[:, 1],
        "l_c": pareto_individui[:, 2],
        "r_c": pareto_individui[:, 3],
        "f1_volume": pareto_funzioni_obiettivo[:, 0],
        "f2_freq": pareto_funzioni_obiettivo[:, 1],
    })

    # Salvataggio CSV (ordinato rispetto al volume)
    df_pareto = df_pareto.sort_values(by = "f1_volume", ascending = True).reset_index(drop=True)
    df_pareto.to_csv(os.path.join(cartella, f"pareto_solutions.csv"), index=False)



##############################################################################################
# Calcolo medie
media_n_pareto = round(np.mean(risultati_n_pareto))
media_n_pareto_filtrato = round(np.mean(risultati_n_pareto_filtrato))
media_hypervolume = np.mean(risultati_hypervolume)


# Stampe
print(f"\n\nVALORI MEDI METRICHE")
print(f"Soluzioni sul fronte: \t\t {media_n_pareto}")
print(f"Soluzioni sul fronte filtrato: {media_n_pareto_filtrato}")
print(f"Ipervolume nella zona d'interesse: \t {media_hypervolume:.6e}")
print(f"Soluzioni non sul fronte: \t\t {sol_per_pop - media_n_pareto}")

###########################################################################################################################
# Salvo CSV per confronti nel tuning
csv_path = os.path.join(cartella_generale, f"risultati_tuning_{tuning}.csv")

def salva_csv_tuning(run_id, p_M_min, parents, K_tournament, n_pareto, n_pareto_filtrato, hypervolume):

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Intestazione
        if not file_exists:
            writer.writerow(["run_id", "p_M_min", "parent_ratio", "K_tournament", "n_pareto", "n_pareto_filtrato", "hypervolume"])

        writer.writerow([run_id, p_M_min, parents, K_tournament, n_pareto, n_pareto_filtrato, hypervolume])

# Salvo CSV
salva_csv_tuning(
    run_id = run_id, 
    p_M_min = iper["p_M_min"], 
    parents = iper["parent_ratio"], 
    K_tournament = K_tournament, 
    n_pareto = media_n_pareto, 
    n_pareto_filtrato = media_n_pareto_filtrato, 
    hypervolume = media_hypervolume
)



