import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qsc import Qsc
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_nor_func_pyqsc import ae_nor_nae
import time

# A configuration is defined with B0, etabar, field periods, Fourier components and eta bar

B0 = 1        # [T] strength of the magnetic field on axis

nfp = 3       # field periods

eta = -0.1    # eta bar

##########################################################################################

# these quantities are provided in [m]

rc = [1, 0.045]
zs = [0, 0.045]

# to calculate things the lam_res needs to be provided, just that?
lam_res = 10000

# distance from the magnetic axis to be used

r = 0.001

a_r = 1

# normalization variable for AE

Delta_r = 1

# gradients for diamagnetic frequency
omn = 0
omt = 0

# gradient array
N = 10
#omn_arr = np.linspace(-20, -1, N)
omt_arr = -1*np.geomspace(1, 100, N)
idx_eta_dagger_arr = np.zeros(N).astype(int)

# arrays for plotting
N_eta = 100
eta_arr = np.geomspace(-700, -0.01, N_eta)

# empty array with shape N by N_eta
ae_total_arr = np.empty((N, N_eta))
ae_nor_total_arr = np.empty((N, N_eta))

eta_arr_4plot = -1*eta_arr

plt.figure(figsize = (12,7))


for idx_omt, omt in enumerate(omt_arr):
    for idx, eta in enumerate(eta_arr):

        stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
        ae_total_arr[idx_omt, idx], ae_nor_total_arr[idx_omt, idx] = ae_nor_nae(stel, r, lam_res, Delta_r, a_r, omn, omt, plot = False)[3:]
  
    idx_eta_dagger_arr[idx_omt] = np.argmax(ae_nor_total_arr[idx_omt, :])

    plt.plot(eta_arr_4plot, ae_nor_total_arr[idx_omt, :], label='omt = {:.2f} ,'.format(omt) + r'$\bar{\eta}_{\dag}$' + ' = {:.3f}'.format(eta_arr[idx_eta_dagger_arr[idx_omt]]))
    plt.scatter(eta_arr_4plot[idx_eta_dagger_arr[idx_omt]], ae_nor_total_arr[idx_omt, idx_eta_dagger_arr[idx_omt]])

plt.title('B0 and axis shape is irrelevant \n r = {}, omn = {}'.format(r, omn), size = 20)
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$-\bar{\eta}$', size = 20)
plt.ylabel(r'$\hat{A} = A/E_t$', size = 20)
plt.legend(loc = 'best', fontsize = 12)
plt.grid(alpha = 0.5)
plt.savefig('ae_nor_eta_dagger_r_{}_omt.png'.format(r), bbox_inches='tight', dpi = 300)
plt.show()

