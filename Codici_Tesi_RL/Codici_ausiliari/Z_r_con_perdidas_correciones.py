import numpy as np
from Codici_ausiliari.crea_Z_k import crea_Z_k


def Z_r_con_perd_cor(neck, cavity, geom):
    
    # Datos
    l_n = neck.l
    dl_n = neck.dl
    l_c = cavity.l
    dl_c = cavity.dl
    dl = dl_n + dl_c
    
    # Calculo Z y k
    Z_n, k_n = crea_Z_k(neck, geom)
    Z_c, k_c = crea_Z_k(cavity, geom)
    

    # Parte imaginaria
    def Z_r_aux(w):
        return -1j * Z_n(w) * (1 - k_n(w)*dl*(Z_n(w)/Z_c(w)*np.tan(k_c(w)*l_c) + np.tan(k_n(w)*l_n)) - Z_n(w)/Z_c(w)*np.tan(k_n(w)*l_n)*np.tan(k_c(w)*l_c)) / (np.tan(k_n(w)*l_n) + k_n(w)*dl*(1 - Z_n(w)/Z_c(w)*np.tan(k_n(w)*l_n)*np.tan(k_c(w)*l_c)) + Z_n(w)/Z_c(w)*np.tan(k_c(w)*l_c))
        #return -1j * Z_n(w) * (1 - k_n(w)*dl*Z_n(w)/Z_c(w)*np.tan(k_c(w)*l_c) - Z_n(w)/Z_c(w)*np.tan(k_n(w)*l_n)*np.tan(k_c(w)*l_c)) / (np.tan(k_n(w)*l_n) - k_n(w)*dl*Z_n(w)/Z_c(w)*np.tan(k_n(w)*l_n)*np.tan(k_c(w)*l_c) + Z_n(w)/Z_c(w)*np.tan(k_c(w)*l_c))
    def Z_r(w):
        return np.imag(Z_r_aux(w))
        
    return Z_r
