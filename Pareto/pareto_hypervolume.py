import pandas as pd
import numpy as np
from pymoo.indicators.hv import Hypervolume

def load_df(path):
    return pd.read_csv(path).sort_values(by='V', ascending=True)

df_dict = {
    '0_128': load_df("Experiment_tuning/tuning_0_128/pareto_0_128.csv"),
    '0_256': load_df("Experiment_tuning/tuning_0_256/pareto_0_256.csv"),
    '0_512': load_df("Experiment_tuning/tuning_0_512/pareto_0_512.csv"),
    '25_128': load_df("Experiment_tuning/tuning_25_128/pareto_25_128.csv"),
    '25_256': load_df("Experiment_tuning/tuning_25_256/pareto_25_256.csv"),
    '25_512': load_df("Experiment_tuning/tuning_25_512/pareto_25_512.csv"),
    '50_128': load_df("Experiment_tuning/tuning_50_128/pareto_50_128.csv"),
    '50_256': load_df("Experiment_tuning/tuning_50_256/pareto_50_256.csv"),
    '50_512': load_df("Experiment_tuning/tuning_50_512/pareto_50_512.csv"),
    '75_128': load_df("Experiment_tuning/tuning_75_128/pareto_75_128.csv"),
    '75_256': load_df("Experiment_tuning/tuning_75_256/pareto_75_256.csv"),
    '75_512': load_df("Experiment_tuning/tuning_75_512/pareto_75_512.csv"),
    '100_128': load_df("Experiment_tuning/tuning_100_128/pareto_100_128.csv"),
    '100_256': load_df("Experiment_tuning/tuning_100_256/pareto_100_256.csv"),
    '100_512': load_df("Experiment_tuning/tuning_100_512/pareto_100_512.csv"),
}

def get_front(percent, df_dict):
    return {
        128: df_dict[f"{percent}_128"],
        256: df_dict[f"{percent}_256"],
        512: df_dict[f"{percent}_512"],
    }

def compute_reference_point(fronts=None):
    return np.array([1e-4, 500])

def compute_hv_for_front(front, ref_point):
    hv_calc = Hypervolume(ref_point=ref_point)
    pts = front[['V', 'w']].values
    hv = hv_calc.do(pts)
    return hv

percentuali = [0, 25, 50, 75, 100]

for perc in percentuali:
    print(f"Percentuale: {perc}%")
    fronts = get_front(perc, df_dict)
    ref_point = compute_reference_point()
    
    hv_scores = {}
    for dim, front in fronts.items():
        hv = compute_hv_for_front(front, ref_point)
        hv_scores[dim] = hv
    
    ranking = sorted(hv_scores.items(), key=lambda x: -x[1])
    
    print("Ranking per Hypervolume:")
    for pos, (dim, score) in enumerate(ranking, 1):
        print(f" {pos}. Dimensione {dim} -> HV = {score:.6f}")
