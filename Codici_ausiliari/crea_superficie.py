import numpy as np


def crea_superficie(oggetto, geom):

    if geom == 1:
        h = oggetto.h
        S = h
        return S 

    elif geom == 2:
        a = oggetto.a
        b = oggetto.b
        S = a*b
        return S

    elif geom == 3:
        r = oggetto.r
        S = np.pi * r**2
        return S

    else:
        raise ValueError(f"The value of geom must be 1, 2 or 3.")

    return S