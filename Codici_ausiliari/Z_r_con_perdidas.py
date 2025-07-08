import numpy as np
from Codici_ausiliari.crea_Z_k import crea_Z_k


def Z_r_con_perd(neck, cavity, geom):
    
    # Datos
    l_n = neck.l
    l_c = cavity.l

    # Calculo Z y k
    Z_n, k_n = crea_Z_k(neck, geom)
    Z_c, k_c = crea_Z_k(cavity, geom)
    

    
    # Parte imaginaria
    def Z_r_aux(w):
        return -1j * Z_n(w) * (Z_c(w)/Z_n(w) - np.tan(k_n(w)*l_n) * np.tan(k_c(w)*l_c)) / (Z_c(w)/Z_n(w) * np.tan(k_n(w)*l_n)+ np.tan(k_c(w)*l_c))

    def Z_r(w):
        return np.imag(Z_r_aux(w))
        
    return Z_r