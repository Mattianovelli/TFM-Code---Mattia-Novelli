import numpy as np
from Codici_ausiliari.Z_k import Z_k


def Z_r_losses(neck, cavity, geom):
    
    # Dati
    l_n     = neck.l
    dl_n    = neck.dl
    l_c     = cavity.l
    dl_c    = cavity.dl
    dl      = dl_n + dl_c
    
    # Calcolo Z e k
    Z_n, k_n = Z_k(neck, geom)
    Z_c, k_c = Z_k(cavity, geom)
    

    # Impedanza e sua parte immaginaria
    def Z_r_comp(w):
        return -1j * Z_n(w) * (1 - k_n(w)*dl*(Z_n(w)/Z_c(w)*np.tan(k_c(w)*l_c) + np.tan(k_n(w)*l_n)) - Z_n(w)/Z_c(w)*np.tan(k_n(w)*l_n)*np.tan(k_c(w)*l_c)) / (np.tan(k_n(w)*l_n) + k_n(w)*dl*(1 - Z_n(w)/Z_c(w)*np.tan(k_n(w)*l_n)*np.tan(k_c(w)*l_c)) + Z_n(w)/Z_c(w)*np.tan(k_c(w)*l_c))
        #return -1j * Z_n(w) * (1 - k_n(w)*dl*Z_n(w)/Z_c(w)*np.tan(k_c(w)*l_c) - Z_n(w)/Z_c(w)*np.tan(k_n(w)*l_n)*np.tan(k_c(w)*l_c)) / (np.tan(k_n(w)*l_n) - k_n(w)*dl*Z_n(w)/Z_c(w)*np.tan(k_n(w)*l_n)*np.tan(k_c(w)*l_c) + Z_n(w)/Z_c(w)*np.tan(k_c(w)*l_c))

    def Z_r_im(w):
        return np.imag(Z_r_comp(w))
        
    return Z_r_im, Z_r_comp