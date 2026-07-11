import numpy as np

# Blend Crossover (SBX)
def blend_crossover(genitori, n_figli, ga_instance):

    # Lunghezza vettori
    n = genitori.shape[1]

    # Creo figli
    figli = np.empty((n_figli[0], n))

    # Conto i genitori
    n_genitori = genitori.shape[0]

    # Estraggo i bounds
    Omega = ga_instance.gene_space
    L = np.array([b['low'] for b in Omega])
    U = np.array([b['high'] for b in Omega])

    # Indice della generazione attuale e numero totale di generazioni
    n_actual_gen = ga_instance.generations_completed   
    n_gen_tot = ga_instance.num_generations

    # Controllo ampiezza intervallo casuale
    r_max = 0.5
    r_min = 0.25
    r = r_max - (r_max - r_min) * (n_actual_gen - 1) / (n_gen_tot - 1)

    # Faccio il crossover lineare (esteso)
    for k in range(n_figli[0]):
        # Campionamento coppie di genitori
        ij = np.random.choice(n_genitori, 2, replace=False)
        g1, g2 = genitori[ij]

        # Valori in [-r, 1 + r] (fa pure estrapolazione)
        alpha = np.random.uniform(-r, 1 + r, size=n)
        #alpha = np.random.normal(0.5, r, size=n)

        # Creazione figlio e lo proiezione su Omega
        figli[k] = alpha * g1 + (1 - alpha) * g2
        figli[k] = np.clip(figli[k], L, U)

    return figli
