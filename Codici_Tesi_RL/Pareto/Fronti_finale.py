import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.ticker import ScalarFormatter

# Caricamento e ordinamento dati
def load_df(path):
    return pd.read_csv(path).sort_values(by='V', ascending=True)

# Caricamento dataset
#df_100_256 = load_df("Experiment_tuning/tuning_100_256/pareto_100_256.csv")
#df_75_256 = load_df("Experiment_tuning/tuning_75_256/pareto_75_256.csv")
df_50_128 = load_df("Experiment_tuning/tuning_50_128/pareto_50_128.csv")
df_25_128 = load_df("Experiment_tuning/tuning_25_128/pareto_25_128.csv")
df_0_128 = load_df("Experiment_tuning/tuning_0_128/pareto_0_128.csv")

# Funzione di plot per una curva
def plot_pareto_lines(df, color, label):
    df_filtered = df[(df['V'] < 1e-4) & (df['w'] < 500)]
    V = df_filtered['V'].values
    w = df_filtered['w'].values

    for i in range(len(V) - 1):
        x0, y0 = V[i], w[i]
        x1, y1 = V[i + 1], w[i + 1]
        if i == 0:
            plt.plot([x0, x1], [y0, y0], color=color, label=label)
        else:
            plt.plot([x0, x1], [y0, y0], color=color)
        plt.plot([x1, x1], [y0, y1], color=color)

    plt.scatter(V, w, color=color, s=30)

# Plot
plt.figure(figsize=(10, 7))

#plot_pareto_lines(df_100_256, 'red', r'$p\% = 100\%,\ n_{\mathrm{neu}}=256$')
#plot_pareto_lines(df_75_256, 'green', r'$p\% = 75\%,\ n_{\mathrm{neu}}=256$')
plot_pareto_lines(df_50_128, 'blue', r'$p\% = 50\%,\ n_{\mathrm{neu}}=128$')
plot_pareto_lines(df_25_128, 'orange', r'$p\% = 25\%,\ n_{\mathrm{neu}}=128$')
plot_pareto_lines(df_0_128, 'purple', r'$p\% = 0\%,\ n_{\mathrm{neu}}=128$')

plt.xlabel('V', fontsize=20)
plt.ylabel(r'$\omega$', fontsize=20)  # Modifica qui per omega
plt.legend(fontsize=18)
plt.grid(True)

# Imposta notazione scientifica con moltiplicatore globale su asse x
ax = plt.gca()
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))  # forza uso della notazione scientifica per piccoli valori
ax.xaxis.set_major_formatter(formatter)

# Tick fontsize
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Limiti
plt.xlim(right=1e-4)
plt.ylim(top=500)

# Salvataggio
folder_name = "Confronto_fronti"
os.makedirs(folder_name, exist_ok=True)
plt.savefig(os.path.join(folder_name, "pareto_confronto_3_fronti.png"))

plt.show()

