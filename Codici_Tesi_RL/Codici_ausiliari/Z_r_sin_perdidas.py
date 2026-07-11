import numpy as np
from Codici_ausiliari.crea_superficie import crea_superficie
from scipy import optimize
from scipy.optimize import root_scalar

def Z_r_sin_perd(neck, cavity, air, geom):
    
    # Datos
    c0 = air.c0
    rho_0 = air.rho_0
    l_n = neck.l
    l_c = cavity.l
    
    
    # Calcolo superficies
    S_n = crea_superficie(neck, geom)
    S_c = crea_superficie(cavity, geom)

    # Calculo impedancias
    Z_n = rho_0 *c0 / S_n
    Z_c = rho_0 * c0 / S_c

    # Z_r sin perdidas y sin correcciones
    def Z_r(w):
        return -Z_n*(Z_c/Z_n - np.tan(w*l_c/c0) * np.tan(w*l_n/c0)) / (Z_c/Z_n * np.tan(w*l_n/c0)+ np.tan(w*l_c/c0))
    return Z_r