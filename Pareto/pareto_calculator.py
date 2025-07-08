## PARETO SET - MATTIA
from paretoset import paretoset
import pandas as pd
import matplotlib.pyplot as plt

def pareto_calculator(dati: pd.DataFrame, paraplot):
    ij = paretoset(dati, sense=["min","min"])
    pareto_front = dati[ij]
    indici = dati[ij].index
    if paraplot == 1:
        no_pareto = dati[~ij]

        plt.figure(figsize=(8, 6))
        plt.scatter(no_pareto.iloc[:, 0], no_pareto.iloc[:, 1], color='blue', alpha=0.5, label='No Pareto points')
        plt.scatter(pareto_front.iloc[:, 0], pareto_front.iloc[:, 1], color='red', alpha=0.9, label='Pareto Front')
        plt.title('Pareto Front')
        plt.legend()
        plt.grid(True)
        plt.savefig("punti_pareto_calculator_risultato.png")
        plt.show()

    return pareto_front, indici