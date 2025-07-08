import numpy as np
from Codici_ausiliari.crea_superficie import crea_superficie
from scipy import optimize
from scipy.optimize import root_scalar
from Codici_ausiliari.Z_r_sin_perdidas import Z_r_sin_perd


def solucion_sin_perdidas(neck, cavity, air, geom, tol, maxit):
    
    # Datos
    c0 = air.c0
    rho_0 = air.rho_0
    l_n = neck.l
    l_c = cavity.l
    
    
    # Calcolo superficies
    S_n = crea_superficie(neck, geom)
    S_c = crea_superficie(cavity, geom)

    # Calculo impedancias
    Z_n = rho_0 * c0 / S_n
    Z_c = rho_0 * c0 / S_c
    
    # Funciones por busqueda numerica (se utiliza solo el numerator)    
    def Z_r_num(w):
        return Z_c / Z_n - np.tan(w * l_c / c0) * np.tan(w * l_n / c0)

    def dZ_r_num(w):
        return -l_c / c0 * ((1 + np.tan(w * l_c / c0))**2) * np.tan(w * l_n / c0) - l_n / c0 * ((1 + np.tan(w * l_n / c0))**2) * np.tan(w * l_c / c0)

    ######################################################
    ######################################################
    # Parametros newton
    if tol is None:
        tol = 1e-6
        
    if maxit is None:
        maxit = 1000

    
    # Punto iniziale
    M = (max(l_c / c0, l_n / c0))
    w_0 = 0.99 * np.pi / (2 * M)
    while(Z_r_num(w_0) > 0):
        w_0 = w_0 + 0.99 * (M - w_0);
    


    # Resoluciòn
    result = root_scalar(Z_r_num, method='newton', fprime=dZ_r_num, x0=w_0, xtol=tol, maxiter=maxit)
    
    if not result.converged:
        raise RuntimeError("Il metodo di Newton non è convergente.")
    sol_sin_perd = result.root


    Z_r = Z_r_sin_perd(neck, cavity, air, geom)
    
    return sol_sin_perd, w_0