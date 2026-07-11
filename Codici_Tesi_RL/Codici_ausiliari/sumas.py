import numpy as np
from Codici_ausiliari.air_features import air_features
from Codici_ausiliari.G import crea_funzioni_G  

def crea_sum_approximations(a,b):



    # Ottieni G_rho e G_K
    G_rho, G_K = crea_funzioni_G()

    def sum_approx_template(G_func, w):
        G2 = G_func(w) ** 2
        suma_app_old = 0
        L = 50

        for m in range(L):
            alpha_m = np.pi / a * (2 * m + 1)
            for n in range(L):
                beta_n = np.pi / b * (2 * n + 1)
                denom = alpha_m**2 * beta_n**2 * (alpha_m**2 + beta_n**2 - G2)
                suma_app_old += 1 / denom

        L += 1
        beta_n = np.pi / b * (2 * (L - 1) + 1)
        suma_app = suma_app_old

        for m in range(L):
            alpha_m = np.pi / a * (2 * m + 1)
            denom = alpha_m**2 * beta_n**2 * (alpha_m**2 + beta_n**2 - G2)
            suma_app += 1 / denom

        for n in range(L - 1):
            beta_n = np.pi / b * (2 * n + 1)
            denom = alpha_m**2 * beta_n**2 * (alpha_m**2 + beta_n**2 - G2)
            suma_app += 1 / denom
        #while (np.abs(suma_app - suma_app_old) / np.abs(suma_app_old) > 1e-7).all():
        while (np.abs(suma_app - suma_app_old) > 1e-15).all():
            L += 1
            print(L)
            beta_n = np.pi / b * (2 * (L - 1) + 1)
            suma_app_old = suma_app

            for m in range(L):
                alpha_m = np.pi / a * (2 * m + 1)
                denom = alpha_m**2 * beta_n**2 * (alpha_m**2 + beta_n**2 - G2)
                suma_app += 1 / denom

            for n in range(L - 1):
                beta_n = np.pi / b * (2 * n + 1)
                denom = alpha_m**2 * beta_n**2 * (alpha_m**2 + beta_n**2 - G2)
                suma_app += 1 / denom

        return suma_app

    def sum_approx_rho(w):
        return sum_approx_template(G_rho, w)

    def sum_approx_K(w):
        return sum_approx_template(G_K, w)

    return sum_approx_rho, sum_approx_K