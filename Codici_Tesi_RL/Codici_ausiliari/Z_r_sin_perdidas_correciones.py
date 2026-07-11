import numpy as np
from Codici_ausiliari.crea_superficie import crea_superficie
from scipy import optimize
from scipy.optimize import root_scalar

def Z_r_sin_perd_cor(neck, cavity, air, geom):
    
    # Datos
    c0 = air.c0
    rho_0 = air.rho_0
    l_n = neck.l
    l_c = cavity.l
    dl_n = neck.dl
    dl_c = cavity.dl
    dl = dl_n + dl_c
    
    # Calcolo superficies
    S_n = crea_superficie(neck, geom)
    S_c = crea_superficie(cavity, geom)

    # Calculo impedancias
    Z_n = rho_0 *c0 / S_n
    Z_c = rho_0 * c0 / S_c

    # Coeficientes
    #A = l_n / c0
    #B = l_c / c0
    #D = -Z_n / Z_c
    #C = dl / c0 * D
    A = l_n / c0
    B = l_c / c0
    D = Z_n / Z_c
    C = dl / c0
    
    # Z_r sin perdidas y sin correcciones
    def Z_r_cor(w):
        return -Z_n * (1 - C*w*(D*np.tan(B*w) + np.tan(A*w)) - D*np.tan(A*w)*np.tan(B*w))/(np.tan(A*w) + C*w*(1-D*np.tan(A*w)*np.tan(B*w)) + D*np.tan(B*w))
        #return -Z_n * (1+ C*w*np.tan(B*w) + D*np.tan(A*w)*np.tan(B*w))/(np.tan(A*w) + C*w*np.tan(A*w)*np.tan(B*w) - D*np.tan(B*w))
    return Z_r_cor
