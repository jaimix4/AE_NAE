import sys
import numpy as np
import matplotlib.pyplot as plt
from qsc import Qsc
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_nor_func_pyqsc import ae_nor_nae
from opt_eta_star import set_eta_star
from ae_func_pyqsc import ae_nae
from scipy.optimize import minimize
import time

# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True

# from Rodriguez paper
# a goes from -0.3 to 0.3
a = 0.045

# b goes from -0.06 to 0.06
b = 0.000

rc = np.array([1, a, b])

zs = np.array([0, a, b])

nfp = 3

eta = -0.9

B0 = 1

lam_res = 10000


Delta_r = 1
a_r = 1

omt = 2
omn = 3

r = 0.001

N_a = 500


def ae_nor_function_a(x):

    rc = np.array([1, x[0]])

    zs = np.array([0, x[0]])

    eta = -0.5

    B0 = 1

    nfp = 3

    stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0)

    eta_star = set_eta_star(stel)

    stel.etabar = eta_star

    return ae_nor_nae(stel, r, lam_res, Delta_r, a_r, omn, omt, plot = False)[-1]

def ae_nor_function_a_2(x):

    rc = np.array([1, x[0]])

    zs = np.array([0, x[0]])

    eta = -0.5

    B0 = 1

    nfp = 3

    stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0)

    eta_star = set_eta_star(stel)

    stel.etabar = eta_star

    print(eta_star)

    return ae_nor_nae(stel, r, lam_res, Delta_r, a_r, omn, omt, plot = False)[-1]

a_start = [0.045]
bounds_a = [[0.01, 0.3]]
result = minimize(ae_nor_function_a, a_start, method='L-BFGS-B', bounds = bounds_a) #, options={'disp': False})

solution = result['x']
print(solution)
ae_min = ae_nor_function_a_2(solution)
print(ae_min)


# simple  minimazation, better for just optimizing eta

def ae_total_function_eta(x):

    #return ae_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_psiA, omn, omT, plot = False)[-1]
    return ae_nor_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_r, a_r, omn, omt, plot = False)[-2]


def ae_nor_total_function_eta(x):

    #return ae_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_psiA, omn, omT, plot = False)[-1]
    #print(omn)
    return ae_nor_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_r, a_r, omn, omt, plot = False)[-1]

