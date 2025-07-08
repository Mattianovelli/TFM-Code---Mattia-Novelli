import numpy as np
from scipy import optimize
from scipy.optimize import root_scalar
from Codici_ausiliari.Z_r_con_perdidas import Z_r_con_perd
from Codici_ausiliari.resolucion_sin_perdidas import solucion_sin_perdidas
import matplotlib.pyplot as plt
import os


def solucion_con_perdidas(neck, cavity, air, geom, tol, maxit):
    
   

    # Creo la funcion
    Z_r = Z_r_con_perd(neck, cavity, geom)


    
    ######################################################
    ######################################################
    # Parametros secants method
    if tol is None:
        tol = 1e-6
        
    if maxit is None:
        maxit = 1000

    
    # Puntos iniciales
    x0 = 1
    x1 = 2
 
    while Z_r(x1) > 0:
        print('Dimezzo x0 e x1. ')
        x0 = x0/2
        x1 = x1/2
#####################################################
#####################################################

    # Resoluciòn
    result = root_scalar(Z_r, method='secant', x0 = x0, x1 = x1, xtol=tol, rtol=tol, maxiter=maxit)
    #result = root_scalar(Z_r, method='bisect', bracket=[x0, x1], xtol=tol, maxiter=maxit)
    sol_con_perd = result.root
    
    if not result.converged:
        print('Neck: ', neck)
        print('Cavity: ', cavity)
        w0, _ = solucion_sin_perdidas(neck, cavity, air, geom, tol, maxit)
        x_vals = np.linspace(0, w0, 500)
        y_vals = np.array([Z_r(x) if not np.isnan(Z_r(x)) else np.nan for x in x_vals])
    
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label='Z_r(x)', color='blue')
        plt.axhline(0, color='red', linestyle='--', label='y = 0')
        plt.title("Z_r(x) en [0, w0] (fallo de convergencia)")
        plt.xlabel("x")
        plt.ylabel("Z_r(x)")
        plt.grid(True)
        plt.legend()
        plt.show()
        sol_con_perd = w0
        print("w0 sin pérdidas: ", w0)
        print("El método no es convergente.")
        raise RuntimeError("El método no es convergente.")
    
    
    

    
    return sol_con_perd, x1
