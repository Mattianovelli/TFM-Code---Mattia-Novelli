import numpy as np
from Codici_ausiliari.surface import surface
from scipy import optimize
from scipy.optimize import root_scalar
from Codici_ausiliari.air_features import air_features
def Z_r_no_losses(neck, cavity, geom):

    air = air_features()
    # Datos
    c0 = air.c0
    rho_0 = air.rho_0
    l_n = neck.l
    l_c = cavity.l
    dl_n = neck.dl
    dl_c = cavity.dl
    dl = dl_n + dl_c
    
    # Calcolo superficies
    S_n = surface(neck, geom)
    S_c = surface(cavity, geom)

    # Calculo impedancias
    Z_n = rho_0 * c0 / S_n
    Z_c = rho_0 * c0 / S_c

    # Coefficients
    A = l_n / c0
    B = l_c / c0
    D = Z_n / Z_c
    C = dl / c0
    
    # Z_r without losses
    def Z_r(w):
        return -Z_n*(1-C*w*(D*np.tan(B*w)+np.tan(A*w))-D*np.tan(A*w)*np.tan(B*w))/(np.tan(A*w)+C*w*(1-D*np.tan(A*w)*np.tan(B*w))+D*np.tan(B*w))
    return Z_r