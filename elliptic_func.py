from numba import jit
import numpy as np
from scipy.special import elliprj, elliprd, elliprf
import timeit
from numba import vectorize, float64, float32

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


def ellip_pi_carlson(n, m):
    """
    Return the complete elliptic integral of the third kind.

    Parameters
    ----------
    n : float
        Real part of the parameter.
    m : float
        Real part of the parameter.

    Returns
    -------
    float
        The complete elliptic integral of the third kind.

    Examples
    --------
    >>> ellip_pi_carlson(0.25, 0.5)
    2.1676193607665555

    """
    return elliprf(0, 1 - m, 1) + (n / 3) * elliprj(0, 1 - m, 1, 1 - n)


# uses scipy
def scipy_carlson_elliptic_fun(n,m):

    Rf = elliprf(0, 1-m, 1)
    Rd = elliprd(0, 1-m, 1)
    Rj = elliprj(0, 1 - m, 1, 1 - n)

    K = Rf
    E = Rf - (m/3)*Rd
    PI = Rf + (n/3)*Rj

    return  K, E, PI


### New fast function that will return the 3 elliptical integrals
def carlson_elliptic_fun(n,m):

    return K, E, PI

#@jit(nopython=True)
@vectorize([float64(float64, float64, float64), float32(float32, float32, float32)], nopython = True)
def RF(x,y,z):

    errtol = 3e-2

    #ierr = 0
    #N = len(y)
    # print("test")
    # print(N)

    xn = x#*np.ones(N)
    yn = y
    zn = z#*np.ones(N)

    #epslon = np.ones(N)

    while True:

        mu = ( xn + yn + zn ) / 3.0
        xndev = 2.0 - ( mu + xn ) / mu
        yndev = 2.0 - ( mu + yn ) / mu
        zndev = 2.0 - ( mu + zn ) / mu

        # for idx in range(N):
        #
        #     epslon[idx] = np.maximum( np.abs(xndev[idx]), np.maximum( np.abs(yndev[idx]), np.abs(zndev[idx]) ) )
        epslon = np.maximum( np.abs(xndev), np.maximum( np.abs(yndev), np.abs(zndev) ) )

        #if (epslon < errtol).all():
        if epslon < errtol:
          c1 = 1.0 / 24.0
          c2 = 3.0 / 44.0
          c3 = 1.0 / 14.0
          e2 = xndev * yndev - zndev * zndev
          e3 = xndev * yndev * zndev
          s = 1.0 + ( c1 * e2 - 0.1 - c2 * e3 ) * e2 + c3 * e3
          value = s / np.sqrt (mu)
          return value

        xnroot = np.sqrt(xn)
        ynroot = np.sqrt(yn)
        znroot = np.sqrt(zn)
        lamda = xnroot * ( ynroot + znroot ) + ynroot * znroot
        xn = ( xn + lamda ) * 0.25
        yn = ( yn + lamda ) * 0.25
        zn = ( zn + lamda ) * 0.25



def RD():


    return 1


def RJ():


    return 1

eta = -0.9
r = 0.05


lamb_min = 1/(1 - r*eta)

lamb_max =  1/(1 + r*eta)

lam_res = 5000

lam_arr = np.linspace(lamb_min, lamb_max, lam_res)[1:-1]

ellip_n = ((-1*(1 - (1 + r*eta)*lam_arr))/((1 + r*eta)*lam_arr))

ellip_m = (-1*(1 - (1 + r*eta)*lam_arr))/(2*r*eta*lam_arr)

#numbaRF = numba.vectorize(RF)

print(elliprf(0, 1-ellip_m, 1)[1000])

# print(numbaRF(0*np.ones_like(ellip_m), 1-ellip_m, 1*np.ones_like(ellip_m))[1000])
print(RF(0, 1-ellip_m, 1)[1000])

# print(elliprf(0, 1-ellip_m, 1)[1000])
#
# print(RF(0, 1-ellip_m, 1)[1000])
#
# print(elliprf(0, 1-ellip_m, 1)[1000])
#
# print(RF(0, 1-ellip_m, 1)[1000])

print("The time taken: {}, value"\
.format(timeit.timeit(lambda: elliprf(0, 1-ellip_m, 1), number = 1)))

print("The time taken: {}, value"\
.format(timeit.timeit(lambda: RF(0, 1-ellip_m, 1), number = 1)))
