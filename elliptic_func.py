from numba import jit
import numpy as np
from scipy.special import elliprj, elliprd

@jit(nopython=True)
def my_ellip_k_e(k, tol=1e-10):
    a = np.ones_like(k)
    g = np.sqrt(1-k)
    #this a is square, but it just ones here, so
    # to be fast lets just leave like that
    c = np.sqrt(np.abs(a - g**2))
    n = 0
    sum_th = 0.5*c**2
    while True:
        a, g = (a + g) / 2, np.sqrt(a * g)
        c = (c**2)/(4*a) # old c with new a, perfect
        n = n + 1
        sum_th = sum_th + (2**(n - 1))*c**2
        if (np.abs(a - g) < tol).all():
            K = (np.pi/(2*a))
            print(n)
            return K, K*(1 - sum_th)

def ellip_pi_carlson(n,m):

    return  elliprf(0, 1-m, 1) + (n/3)*elliprj(0, 1 - m, 1, 1 - n)
