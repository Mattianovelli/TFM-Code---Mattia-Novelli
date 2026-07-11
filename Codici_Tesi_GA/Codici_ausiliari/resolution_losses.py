import numpy as np
from scipy import optimize
from scipy.optimize import root_scalar
from Codici_ausiliari.Z_r_losses import Z_r_losses

def resolution_losses(neck, cavity, air, geom, x0, x1, tol, maxit):
    
    # Datos
    c0 = air.c0
    l_n = neck.l
    l_c = cavity.l
    

    # Creo la funcion
    Z_r_im, Z_r_comp = Z_r_losses(neck, cavity, geom)


    
    ######################################################
    ######################################################
    # Parametros secants method
    if tol is None:
        tol = 1e-6
        
    if maxit is None:
        maxit = 1000

    while Z_r_im(x1) > 0:
        #print('Dimezzo x0 e x1. ')
        x0 = x0/2
        x1 = x1/2
    #####################################################
    #####################################################
    #####################################################

    # Resoluciòn
    secant_result = root_scalar(Z_r_im, method='secant', x0 = x0, x1 = x1, xtol=tol, rtol=tol, maxiter=maxit)
    sol_losses = secant_result.root
    if sol_losses < 0:
        sol_losses = np.inf
    #print("La raiz con perdidas y correciones es en w=", sol_con_perd_cor)
    
    return sol_losses

