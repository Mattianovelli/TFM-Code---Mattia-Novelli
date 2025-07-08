import numpy as np
from Codici_ausiliari.crea_superficie import crea_superficie
from Codici_ausiliari.crea_rho_K import crea_rho_K

# Creacion de Z y k con rho(w) y K(w)
def crea_Z_k(oggetto, geom):
    S = crea_superficie(oggetto, geom)
    rho, K = crea_rho_K(oggetto, geom)
    
    def Z(w):
        Z_val = np.sqrt(rho(w)*K(w))/S
        return Z_val

    def k(w):
        k_val = w*np.sqrt(rho(w)/K(w))
        return k_val

    return Z, k