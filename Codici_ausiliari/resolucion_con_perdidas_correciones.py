import numpy as np
from scipy import optimize
from scipy.optimize import root_scalar
from Codici_ausiliari.Z_r_con_perdidas_correciones import Z_r_con_perd_cor

def solucion_con_perdidas_correciones(neck, cavity, air, geom, tol, maxit):
    
    # Datos
    c0 = air.c0
    l_n = neck.l
    l_c = cavity.l
    

    # Creo la funcion
    Z_r = Z_r_con_perd_cor(neck, cavity, geom)


    
    ######################################################
    ######################################################
    # Parametros secants method
    if tol is None:
        tol = 1e-6
        
    if maxit is None:
        maxit = 1000

    x0 = 1
    x1 = 2
    while Z_r(x1) > 0:
        print('Dimezzo x0 e x1. ')
        x0 = x0/2
        x1 = x1/2
    #####################################################
    #####################################################
    #####################################################

    # Resoluciòn
    secant_result = root_scalar(Z_r, method='secant', x0 = x0, x1 = x1, xtol=tol, rtol=tol, maxiter=maxit)
    sol_con_perd_cor = secant_result.root
    #print("La raiz con perdidas y correciones es en w=", sol_con_perd_cor)
    
    return sol_con_perd_cor, x1