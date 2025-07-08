import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.ticker import ScalarFormatter

# Caricamento e ordinamento dati
def load_df(path):
    return pd.read_csv(path).sort_values(by='V', ascending=True)

# Caricamento dataset
df_sin_finali = load_df("Experiment_sin_correciones_150/exp_sin_cor_25_128/valori_finali_25_128.csv")
df_con_finali = load_df("Experiment_con_correciones_1150/exp_con_cor_25_128/valori_finali_cor_25_128.csv")
df_dob_con_finali = load_df("Experiment_con_doble_correciones_150/exp_con_dob_cor_25_128_150/valori_finali_dob_cor_25_128_150.csv")
df_sin = load_df("Experiment_sin_correciones_150/exp_sin_cor_25_128/pareto_25_128.csv")
df_con = load_df("Experiment_con_correciones_150/exp_con_cor_25_128/pareto_25_128.csv")
df_dob_con = load_df("Experiment_con_doble_correciones_150/exp_con_dob_cor_25_128_150/pareto_25_128.csv")


experiment_values = df_sin.iloc[:, 0].unique()
df_filtered_sin = df_sin_finali[df_sin_finali['Experiment'].isin(experiment_values)]
os.makedirs("Tabelle_geometria", exist_ok=True)
df_filtered_sin.to_csv("Tabelle_geometria/filtered_experiments_sin.csv", index=False)




experiment_values = df_con.iloc[:, 0].unique()
df_filtered_con = df_con_finali[df_con_finali['Experiment'].isin(experiment_values)]
os.makedirs("Tabelle_geometria", exist_ok=True)
df_filtered_con.to_csv("Tabelle_geometria/filtered_experiments_con.csv", index=False)

experiment_values = df_dob_con.iloc[:, 0].unique()
df_filtered_dob_con = df_dob_con_finali[df_dob_con_finali['Experiment'].isin(experiment_values)]
os.makedirs("Tabelle_geometria", exist_ok=True)
df_filtered_dob_con.to_csv("Tabelle_geometria/filtered_experiments_dob_con.csv", index=False)





