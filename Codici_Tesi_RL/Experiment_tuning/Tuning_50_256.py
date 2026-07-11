import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv
from dataclasses import dataclass
#import sys
#sys.path.append('C:/Users/matti/Documents/Uni - Valencia/TFM/Codici ausiliari')
from Codici_ausiliari.air_features import air_features
#from Codici_ausiliari.resolucion_sin_perdidas import solucion_sin_perdidas
from Codici_ausiliari.resolucion_sin_perdidas_correciones import solucion_sin_perdidas_correciones
#from Codici_ausiliari.resolucion_con_perdidas import solucion_con_perdidas
from Codici_ausiliari.resolucion_con_perdidas_correciones import solucion_con_perdidas_correciones
#from Codici_ausiliari.Z_r_sin_perdidas import Z_r_sin_perd
#from Codici_ausiliari.Z_r_sin_perdidas_correciones import Z_r_sin_perd_cor
#from Codici_ausiliari.Z_r_con_perdidas import Z_r_con_perd
#from Codici_ausiliari.Z_r_con_perdidas_correciones import Z_r_con_perd_cor
from pareto_calculator import pareto_calculator
import pandas as pd

# nohup python3 Tuning_50_256.py > Tuning_50_256.out &

## Parametros 
perc=0.5;
dim1 = 256;
dim2 = dim1;
air = air_features()
maxit = 1000;
tol = 1e-8
maxit = 1000;
geom = 3;
c0 = air.c0
paso = 0.001

# Calcolo del valore in percentuale intero
perc100 = int(perc * 100)

# Cartella principale
base_dir = "Experiment_tuning"
os.makedirs(base_dir, exist_ok=True)

# Sottocartella per questa percentuale
exp_subdir = os.path.join(base_dir, f"tuning_{perc100}_{dim1}")
os.makedirs(exp_subdir, exist_ok=True)

# Sottocartella dei grafici 
grafici_dir = os.path.join(exp_subdir, f"grafici_{perc100}_{dim1}")
os.makedirs(grafici_dir, exist_ok=True)


# File CSV
csv_completo = os.path.join(exp_subdir, f"valori_completi_{perc100}_{dim1}.csv")
with open(csv_completo, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "Iteration", "l_c", "r_c", "l_n", "r_n", "V", "w"])

csv_finale = os.path.join(exp_subdir, f"valori_finali_{perc100}_{dim1}.csv")
with open(csv_finale, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "l_c", "r_c", "l_n", "r_n", "V", "w"])

csv_best = os.path.join(exp_subdir, f"best_{perc100}_{dim1}.csv")
with open(csv_best, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["w_min", "f_min", "V_min", "w_opt", "f_opt", "V_opt", "iteration"])




# --------------------
# Ambiente
# --------------------

class Environment:
    def __init__(self):
        self.state0 = np.array([0.04, 0.02, 0.04, 0.02]) # [l_c, r_c, l_n, r_n]
        self.lim_min = np.array([0.01, 0.001, 0.01, 0.001])
        self.lim_max = np.array([0.08, 0.04, 0.08, 0.04])
        self.reset()

    def reset(self, rand = False):
        if np.random.rand() < perc:
            #self.state = np.array([np.random.uniform(paso, self.state0[0]*2),
                                   #np.random.uniform(paso, self.state0[1]*2),
                                   #np.random.uniform(paso, self.state0[2]*2),
                                   #np.random.uniform(paso, self.state0[3]*2)])
            self.state = np.array([np.random.uniform(self.lim_min[0], self.lim_max[0]),
                                   np.random.uniform(self.lim_min[1], self.lim_max[1]),
                                   np.random.uniform(self.lim_min[2], self.lim_max[2]),
                                   np.random.uniform(self.lim_min[3], self.lim_max[3])])
        else:
            self.state = self.state0
        self.V = self.compute_V(self.state)
        self.w = self.compute_w(self.state)
        return self.state.copy(), self.V, self.w

    def compute_V(self, s):
        l_c, r_c, l_n, r_n = s
        return np.pi*(r_c**2 * l_c + r_n**2 * l_n)


    def compute_w(self, s):
        l_c, r_c, l_n, r_n = s
        return c0 * np.sqrt((r_n**2) / (l_n * l_c * r_c**2))

    def step(self, action_idx):
        actions = [
            (0, +paso), (0, -paso),
            (1, +paso), (1, -paso),
            (2, +paso), (2, -paso),
            (3, +paso), (3, -paso),
            (0 ,0)
        ]

        prev_V = self.V
        prev_w = self.w

        var_idx, delta = actions[action_idx]
        new_state = self.state.copy()
        new_state[var_idx] += delta
        new_state[var_idx] = np.clip(new_state[var_idx], self.lim_min[var_idx], self.lim_max[var_idx])  # evita divisioni per 0

        self.state = new_state
        self.V = self.compute_V(self.state)
        self.w = self.compute_w(self.state)

        reward = self.get_reward(prev_V, self.V, prev_w, self.w)

        return self.state.copy(), self.V, self.w, reward

    def get_reward(self, prev_V, curr_V, prev_w, curr_w):
        def delta(old, new):
            if abs(new - old)/abs(old) < 1e-4:
                return "="
            elif new < old:
                return "↓"
            else:
                return "↑"

        v_change = delta(prev_V, curr_V)
        w_change = delta(prev_w, curr_w)

        reward_table = {
            ("↓", "↓"): 5,
            ("↓", "="): 2,
            ("↓", "↑"): -1,
            ("=", "↓"): 4,
            ("=", "="): 0,
            ("=", "↑"): -5,
            ("↑", "↓"): 0,
            ("↑", "="): -3,
            ("↑", "↑"): -6,
        }

        return reward_table[(v_change, w_change)]

# --------------------
# Agente
# --------------------
class Agent:
    def __init__(self, lr=2e-3, gamma=0.9, epsilon=1.0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.999
        self.batch_size = 32
        self.memory = deque(maxlen=100000)

        
        self.model = nn.Sequential(
            nn.Linear(4, dim1),
            nn.ReLU(),
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Linear(dim2 , 9)
        )
        for i in self.model:
            if not isinstance(i, nn.Linear):
                continue
            nn.init.constant_(i.weight, 0.0)
            nn.init.constant_(i.bias, 0.0)
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 7)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def store(self, transition):
        self.memory.append(transition)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        kk = list(range(len(self.memory)))
        random.shuffle(kk)
        
        memory2 = [self.memory[r] for r in kk]
        
        for i in range(0, len(memory2), self.batch_size):
            batch = memory2[i:i + self.batch_size]
            states, actions, rewards, next_states = zip(*batch)
            states = torch.FloatTensor(np.array(states))
            next_states = torch.FloatTensor(np.array(next_states))
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            actions = torch.LongTensor(actions).unsqueeze(1)

            next_q_values = self.model(next_states).max(1)[0].detach().unsqueeze(1)
            self.optimizer.zero_grad()
            q_values = self.model(states).gather(1, actions)
            target = rewards + self.gamma * next_q_values

            loss = nn.MSELoss()(q_values, target)
            loss.backward()
            self.optimizer.step()

# --------------------
# --------------------
env = Environment()
agent = Agent()

resultados_finales = []
V_history = []
w_history = []
n_experiments = 150;
n_iterations = 100;

for j in tqdm(range(n_experiments)):
    state, V, w = env.reset()
    history_states = []
  
    for i in range(n_iterations):
        input_state = np.array(state)
        action = agent.select_action(input_state)
        next_state, next_V, next_w, reward = env.step(action)
        next_input = np.array(next_state)

        agent.store((input_state, action, reward, next_input))
        agent.train()

        # Salva dati completi a ogni iterazione
        l_c, r_c, l_n, r_n = state
        with open(csv_completo, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([j, i, l_c, r_c, l_n, r_n, V, w])

        state, V, w = next_state, next_V, next_w
        V_history.append(V)
        w_history.append(w)
        history_states.append(state.copy())

    resultados_finales.append({
        "Experimento": j,
        "V": V,
        "w": w,
        "Estado": state,
    })

    # Salva dati finali dell’esperimento
    l_c, r_c, l_n, r_n = state
    with open(csv_finale, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([j, l_c, r_c, l_n, r_n, V, w])

    if j % 5 == 0:
        history_states = np.array(history_states)
        plt.figure(figsize=(10, 6))
        plt.plot(history_states[:, 0], label="l_c")
        plt.plot(history_states[:, 1], label="r_c")
        plt.plot(history_states[:, 2], label="l_n")
        plt.plot(history_states[:, 3], label="r_n")
        plt.xlabel("Iterazioni interne")
        plt.ylabel("Valore variabile")
        plt.title(f"Andamento variabili per esperimento {j}/{n_experiments}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(grafici_dir, f"variabili_esperimento_{j}.png"), dpi=300)
        #plt.show()
        plt.close()


####################################################################################
###############################################################
# --------------------
# --------------------
V_vals = np.array([r["V"] for r in resultados_finales])
w_vals = np.array([r["w"] for r in resultados_finales])




# Trova il punto più vicino al minimo
V_min = min(V_vals)
w_min = min(w_vals)

norms = np.sqrt((w_vals-w_min)**2 + (V_vals-V_min)**2)
ind = np.argmin(norms)

w_opt = w_vals[ind]
V_opt = V_vals[ind]
f_min = w_min / (2 * np.pi)
f_opt = w_opt / (2 * np.pi)

print('Volumen minimo: ', V_min, ', Frecuencia angular minima: ', w_min, ', Frecuencia minima: ', f_min)
print('Volumen optimo: ', V_opt, ', Frecuencia angular minima: ', w_opt, ', Frecuencia minima: ', f_opt)

with open(csv_best, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([w_min, f_min, V_min, w_opt, f_opt, V_opt, ind])

## Disegno finale dei risultati
experiment_ids = [r["Experimento"] for r in resultados_finales]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(V_vals, w_vals, c=experiment_ids, cmap='viridis', s=50, alpha=0.8)
plt.xlabel("Volume V")
plt.ylabel("Frequenza w")
plt.title("Distribuzione dei risultati nel piano V-w (colorati per esperimento)")
plt.colorbar(scatter, label="Numero Esperimento")
plt.grid(True)

# Testo con i valori chiave
perc_text = f"Percentuale randomizzazione: {perc * 100:.1f}%"
info_text = (
    f"$w_{{min}}$ = {w_min:.3f}, $f_{{min}}$ = {f_min:.3f} Hz, $V_{{min}}$ = {V_min:.2e}\n"
    f"$w_{{opt}}$ = {w_opt:.3f}, $f_{{opt}}$ = {f_opt:.3f} Hz, $V_{{opt}}$ = {V_opt:.2e}\n"
    f"{perc_text}"
)
plt.gcf().text(0.5, -0.12, info_text, fontsize=10, ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()  
plt.savefig(os.path.join(grafici_dir, f"risultati_V_w_{perc100}.png"), dpi=300, bbox_inches='tight')
#plt.show()
plt.close()


###############################################################
###############################################################
###############################################################
# Calcolo del Pareto Front
# Costruisci DataFrame con w e V
df_pareto = pd.DataFrame({
    "w": w_vals,
    "V": V_vals
})

# Calcolo fronte
pareto_front, index_pareto = pareto_calculator(df_pareto, paraplot=0)

# Creo CSV
csv_pareto = os.path.join(exp_subdir, f"pareto_{perc100}_{dim1}.csv")
pareto_front.to_csv(csv_pareto, index=True)

# Disegno del fronte di Pareto
plt.figure(figsize=(8, 6))
plt.scatter(df_pareto["V"], df_pareto["w"], color='blue', alpha=0.5, label='All the points')
plt.scatter(pareto_front["V"], pareto_front["w"], color='red', alpha=0.9, label='Pareto Front')
plt.xlabel(r"Volume $V$")
plt.ylabel(r"Angular Frequency $\omega$")
plt.title(r"Pareto Front - Root approximation of $\omega_r$")
plt.legend()
plt.grid(True)

# Testo dati problema
info_text = f"$dim_1$ = {dim1}, $dim_2$ = {dim2}, Randomness = {perc100}%"
plt.gcf().text(0.95, 0.05, info_text, fontsize=10, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7))

# Salva immagine
pareto_plot_path = os.path.join(grafici_dir, f"pareto_front_{perc100}_{dim1}_{dim2}.png")
plt.tight_layout()
plt.savefig(pareto_plot_path, dpi=300)
#plt.show()
plt.close()

