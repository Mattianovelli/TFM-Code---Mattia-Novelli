import numpy as np

def gaussian_mutation_mia(offspring, ga_instance):
    sigma = 0.01

    Omega = ga_instance.gene_space
    L = np.array([b['low'] for b in Omega])
    U = np.array([b['high'] for b in Omega])
    
    p_M = ga_instance.mutation_probability
    
    # Decido se mutare il figlio
    mutation_mask = np.random.rand(*offspring.shape) < p_M

    # Passo gaussiano
    noise = np.random.normal(0, sigma, size=offspring.shape)

    # Mutazione
    offspring[mutation_mask] += noise[mutation_mask]
    offspring = np.clip(offspring, L, U)
    
    return offspring