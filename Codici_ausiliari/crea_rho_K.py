import numpy as np
from scipy.special import jv
from Codici_ausiliari.G import crea_funzioni_G
from Codici_ausiliari.sumas import crea_sum_approximations
from Codici_ausiliari.air_features import air_features
from Codici_ausiliari.resolucion_sin_perdidas import solucion_sin_perdidas
air = air_features()

def crea_rho_K(oggetto, geom):
    
    # Parametri dell’aria
    rho_0 = air.rho_0
    K0 = air.K0
    gamma = air.gamma

    # Funzioni G_rho e G_K
    G_rho, G_K = crea_funzioni_G()

    if geom == 1:
        h = oggetto.h
        def rho(w):
            h2G = (h / 2) * G_rho(w)
            return rho_0 * (1 - np.tanh(h2G) / h2G)**(-1)

        def K(w):
            h2G = (h / 2) * G_K(w)
            return K0 * (1 + (gamma - 1) * np.tanh(h2G) / h2G)**(-1)

    elif geom == 2:
        a = oggetto.a
        b = oggetto.b
        sum_approx_rho, sum_approx_K = crea_sum_approximations(a, b)
        def rho(w):
            suma = sum_approx_rho(w)
            A = (a * b / 4)
            return -rho_0 * A**2 / (4 * G_rho(w)**2 * suma)

        def K(w):
            suma = sum_approx_K(w)
            A = (a * b / 4)
            return K0 / (gamma + 4 * (gamma - 1) * G_K(w)**2 / A**2 * suma)

    elif geom == 3:
        r = oggetto.r
        #def rho(w):
        #    rG = r * G_rho(w)
        #    return rho_0 * (1 - 2 / rG * jv(1, rG) / jv(0, rG))**(-1)

        #def K(w):
        #    rG = r * G_K(w)
        #    return K0 * (1 + 2 * (gamma - 1) / rG * jv(1, rG) / jv(0, rG))**(-1)

        def rho(w):
            rG = r * G_rho(w)
            with np.errstate(all='ignore'):
                val = rho_0 * (1 - 2 / rG * jv(1, rG) / jv(0, rG))**(-1)
            if not np.isfinite(val):
                print('Problema con rho. w = ', w, ', rG = ', rG)
                return rho_0
            return val

        def K(w):
            rG = r * G_K(w)
            with np.errstate(all='ignore'):
                val = K0 * (1 + 2 * (gamma - 1) / rG * jv(1, rG) / jv(0, rG))**(-1)
            if not np.isfinite(val):
                print('Problema con K. w = ', w, ', rG = ', rG)
                return K0
            return val
    else:
        raise ValueError(f"The value of geom must be 1, 2 or 3.")
    
    return rho, K