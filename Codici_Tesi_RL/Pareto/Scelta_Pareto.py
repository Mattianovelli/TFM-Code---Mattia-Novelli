import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.ticker import ScalarFormatter

# Caricamento e ordinamento dati
def load_df(path):
    return pd.read_csv(path).sort_values(by='V', ascending=True)

df_100_512 = load_df("Experiment_tuning/tuning_100_512/pareto_100_512.csv")
df_100_256 = load_df("Experiment_tuning/tuning_100_256/pareto_100_256.csv")
df_100_128 = load_df("Experiment_tuning/tuning_100_128/pareto_100_128.csv")
df_75_512 = load_df("Experiment_tuning/tuning_75_512/pareto_75_512.csv")
df_75_256 = load_df("Experiment_tuning/tuning_75_256/pareto_75_256.csv")
df_75_128 = load_df("Experiment_tuning/tuning_75_128/pareto_75_128.csv")
df_50_512 = load_df("Experiment_tuning/tuning_50_512/pareto_50_512.csv")
df_50_256 = load_df("Experiment_tuning/tuning_50_256/pareto_50_256.csv")
df_50_128 = load_df("Experiment_tuning/tuning_50_128/pareto_50_128.csv")
df_25_512 = load_df("Experiment_tuning/tuning_25_512/pareto_25_512.csv")
df_25_256 = load_df("Experiment_tuning/tuning_25_256/pareto_25_256.csv")
df_25_128 = load_df("Experiment_tuning/tuning_25_128/pareto_25_128.csv")
df_0_512 = load_df("Experiment_tuning/tuning_0_512/pareto_0_512.csv")
df_0_256 = load_df("Experiment_tuning/tuning_0_256/pareto_0_256.csv")
df_0_128 = load_df("Experiment_tuning/tuning_0_128/pareto_0_128.csv")

# Funzione di plotting
def plot_group(df_512, df_256, df_128, p_percent):
    plt.figure(figsize=(8, 6))

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

    plot_pareto_lines(df_512, 'red', fr'$p\% = {p_percent}\%,\ n_{{\mathrm{{neu}}}}=512$')
    plot_pareto_lines(df_256, 'green', fr'$p\% = {p_percent}\%,\ n_{{\mathrm{{neu}}}}=256$')
    plot_pareto_lines(df_128, 'blue', fr'$p\% = {p_percent}\%,\ n_{{\mathrm{{neu}}}}=128$')

    plt.xlabel('V', fontsize=16)
    plt.ylabel(r'$\omega$', fontsize=16)   # Modifica qui per omega
    plt.title(fr'Pareto Front:  $p\% = {p_percent}\%$', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)

    # Notazione scientifica compatta per asse x
    ax = plt.gca()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(formatter)

    # Font tick
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlim(right=1e-4)
    plt.ylim(top=500)

    # Salvataggio
    folder_name = "Confronto_fronti"
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(os.path.join(folder_name, f"pareto_plot_{p_percent}.png"))

    plt.show()

# Esegui per ogni p%
plot_group(df_100_512, df_100_256, df_100_128, 100)
plot_group(df_75_512, df_75_256, df_75_128, 75)
plot_group(df_50_512, df_50_256, df_50_128, 50)
plot_group(df_25_512, df_25_256, df_25_128, 25)
plot_group(df_0_512, df_0_256, df_0_128, 0)
