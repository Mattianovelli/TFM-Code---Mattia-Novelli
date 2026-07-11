## Z(w) e k(w)

import numpy as np
from Codici_ausiliari.surface import surface
from Codici_ausiliari.rho_K import rho_K

# Creazione di Z e k con rho(w) e K(w)
def Z_k(oggetto, geom):
    S = surface(oggetto, geom)
    rho, K = rho_K(oggetto, geom)
    
    def Z(w):
        Z_val = np.sqrt(rho(w) * K(w)) / S
        return Z_val

    def k(w):
        k_val = w * np.sqrt(rho(w) / K(w))
        return k_val

    return Z, k