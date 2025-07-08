import numpy as np
from Codici_ausiliari.crea_superficie import crea_superficie
from scipy import optimize
from scipy.optimize import root_scalar

def solucion_sin_perdidas_correciones(neck, cavity, air, geom, tol, maxit):
    
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
    
    # Funciones por busqueda numerica (se utiliza solo el numerator)    
    def Z_r_cor_den(x):
        return 1 - C*x*(D*np.tan(B*x) + np.tan(A*x)) - D*np.tan(A*x)*np.tan(B*x)
        #return 1+ C*x*np.tan(B*x) + D*np.tan(A*x)*np.tan(B*x)
    def dZ_r_cor_den(x):
        tanA = np.tan(A*x)
        tanB = np.tan(B*x)
    
        dtanA = 1 + tanA**2  
        dtanB = 1 + tanB**2  

        termine_1 = -C * (D*tanB + tanA)
        termine_2 = -C * x * (D*B*dtanB + A*dtanA)
        termine_3 = -D * (A*dtanA*tanB + B*dtanB*tanA)
        return termine_1 + termine_2 + termine_3
        #return  C*(np.tan(B*x) + B*x*(1+np.tan(B*x)**2)) + D*(A*(1+np.tan(A*x)**2)*np.tan(B*x)+B*np.tan(A*x)*(1+np.tan(B*x)**2))

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
    while(Z_r_cor_den(w_0) > 0):
        w_0 = w_0 + 0.99 * (M - w_0);
  

    # Resoluciòn
    sol_sin_perd_cor = optimize.newton(Z_r_cor_den, w_0, fprime = dZ_r_cor_den, tol = tol, maxiter = maxit)
    
    return sol_sin_perd_cor, w_0