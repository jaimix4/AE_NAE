import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qsc import Qsc
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_func_pyqsc import ae_nae
import time

# A configuration is defined with B0, etabar, field periods, Fourier components and eta bar

B0 = 1         # [T] strength of the magnetic field on axis

eta = -0.28484723      # parameter

nfp = 3         # number of field periods

rc=[1, 0.045]   # rc -> the cosine components of the axis radial component
zs=[0, 0.045]  # zs -> the sine components of the axis vertical component

rc=[1, 0.06961988, -0.04796981]   # rc -> the cosine components of the axis radial component
zs=[0, 0.06961988, -0.04796981]  # zs -> the sine components of the axis vertical component


# these quantities are provided in [m]

# this is the configuration of r1 section 5.1 from
# (not sure) Landreman, Sengupta, and Plunk, Journal of Plasma Physics 85, 905850103 (2019).

# stel_1 = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=-2.5, B0=B0)
# stel_2 = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=-0.9, B0=B0)
# stel_3 = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=-0.2, B0=B0)

# to calculate things the lam_res needs to be provided, just that?
lam_res = 5000

# distance from the magnetic axis to be used
r = 0.005

# stel_1.plot_boundary(r = r)
# stel_2.plot_boundary(r = r)
# stel_3.plot_boundary(r = r)

# normalization variable for AE
Delta_psiA = 1

# gradients for diamagnetic frequency
dln_n_dpsi = -1
dln_T_dpsi = 0

# 1 or 3 gradients
omn = dln_n_dpsi
omT = dln_T_dpsi

stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
stel.plot_boundary(r = r)

### Optimization stuff

from scipy.optimize import minimize


def ae_total_function_eta(x):

    return ae_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_psiA, omn, omT, plot = False)[-1]

eta_min = -4

eta_max = -0.01

eta_start = [-0.8]

# simple  minimazation, better for just optimizing eta
result = minimize(ae_total_function_eta, eta_start, method='L-BFGS-B')

# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = ae_total_function_eta(solution)
print(r)
print(solution)
print(evaluation)

stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=solution[0], B0=B0)
stel.plot_boundary(r = r)


# global Optimization

# simulated annealing global optimization for a multimodal objective function
from scipy.optimize import dual_annealing

def ae_total_function(x):

    return ae_nae(Qsc(rc=np.array([1, x[0], x[1]]), zs=np.array([0, x[0], x[1]]), nfp=nfp, etabar=x[2], B0=B0), r, lam_res, Delta_psiA, omn, omT, plot = False)[-1]

# constrains

# define range for input
# define the bounds on the search
bounds = [[0.001, 0.3], [-0.06, 0.06], [-1.2, -0.1]]

# perform the simulated annealing search
result = dual_annealing(ae_total_function, bounds)

# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = ae_total_function(solution)
print(r)
print(solution)
print(evaluation)

stel = Qsc(rc=[1, solution[0], solution[1]], zs=[0, solution[0], solution[1]], nfp=nfp, etabar=solution[2], B0=B0)
stel.plot_boundary(r = r)
