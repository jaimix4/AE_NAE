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
from   matplotlib        import rc
# add latex fonts to plots
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 22})
rc('text', usetex=True)



# A configuration is defined with B0, etabar, field periods, Fourier components and eta bar

B0 = 1        # [T] strength of the magnetic field on axis

nfp = 3       # field periods

eta = -5   # eta bar

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
omn = 3
omt = 3

# gradient array
N = 5
#omn_arr = np.linspace(-20, -1, N)
# omn_arr = np.geomspace(1, 10, N)
omn_arr = [0.5, 1, 2, 5, 10]

# omt_arr = np.geomspace(1, 10, N)
omt_arr = [0.5, 1, 2, 5, 10]

idx_eta_dagger_arr = np.zeros(N).astype(int)

# arrays for plotting
N_eta = 100
eta_arr = np.geomspace(-700, -0.01, N_eta)

# arrays of r
N_r = 50
r_arr = np.geomspace(1e-12, 1e-3, N_r)

# empty array with shape N by N_eta
ae_total_arr = np.empty((N, N_r))
ae_nor_total_arr = np.empty((N, N_r))

eta_arr_4plot = -1*eta_arr


plt.figure(figsize = (12/1.5,7.5/1.5))


for idx_omn, omn in enumerate(omn_arr):
    for idx, r in enumerate(r_arr):

        stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
        ae_total_arr[idx_omn, idx], ae_nor_total_arr[idx_omn, idx] = ae_nor_nae(stel, r, lam_res, Delta_r, a_r, omn, omt, plot = False)[3:]
  
    #idx_eta_dagger_arr[idx_omt] = np.argmax(ae_nor_total_arr[idx_omt, :])

    plt.plot(r_arr, ae_nor_total_arr[idx_omn, :], linewidth = 4, label=r'$\hat{\omega}_{n}$' + ' = {:.2f} '.format(omn))# + r'$\bar{\eta}_{\dag}$' + ' = {:.3f}'.format(eta_arr[idx_eta_dagger_arr[idx_omt]]))
    
    # calculate slope of log(r_arr) and log(ae_nor_total_arr)
    slope, intercept = np.polyfit(np.log10(r_arr), np.log10(ae_nor_total_arr[idx_omn, :]), 1)
    print(slope)

    # figure power law
    #plt.plot(r_arr, 1e-3*r_arr**(-1.5), '--', linewidth = 4, label=r'$\hat{\omega}_{n}$' + ' = {:.2f} '.format(omn))# + r'$\bar{\eta}_{\dag}$' + ' = {:.3f}'.format(eta_arr[idx_eta_dagger_arr[idx_omt]]))
    
    #plt.scatter(eta_arr_4plot[idx_eta_dagger_arr[idx_omt]], ae_nor_total_arr[idx_omt, idx_eta_dagger_arr[idx_omt]])

#plt.title('$B_0$ and axis shape is irrelevant \n' + r'$\hat{\omega}_{T}$' + ' = {} , '.format(omt) + r'$\bar{\eta}$' + ' = {}'.format(eta), size = 18)

# plt.axis([0, 10, 0, 10])
plt.text(2e-6, 2e-3, r'$\hat{\omega}_{T}$' + ' = {} , '.format(omt) + r'$\bar{\eta}$' + ' = {}'.format(eta))

# figure power law


plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$r$', size = 28)
plt.ylabel(r'$\hat{A}$', size = 28)
plt.legend(loc = 'best', fontsize = 20)
plt.grid(alpha = 0.5)
plt.savefig('THESIS_r_ae_nor_eta_{}.png'.format(-eta), bbox_inches='tight', dpi = 300)
plt.show()

plt.figure(figsize = (12/1.5,8/1.5))

omn = 0
omt = 0

for idx_omn, omt in enumerate(omt_arr):
    for idx, r in enumerate(r_arr):

        stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
        ae_total_arr[idx_omn, idx], ae_nor_total_arr[idx_omn, idx] = ae_nor_nae(stel, r, lam_res, Delta_r, a_r, omn, omt, plot = False)[3:]
  
    #idx_eta_dagger_arr[idx_omt] = np.argmax(ae_nor_total_arr[idx_omt, :])

    plt.plot(r_arr, ae_nor_total_arr[idx_omn, :], linewidth = 4, label=r'$\hat{\omega}_{T}$' + ' = {:.2f} '.format(omt))# + r'$\bar{\eta}_{\dag}$' + ' = {:.3f}'.format(eta_arr[idx_eta_dagger_arr[idx_omt]]))
    #plt.scatter(eta_arr_4plot[idx_eta_dagger_arr[idx_omt]], ae_nor_total_arr[idx_omt, idx_eta_dagger_arr[idx_omt]])

#plt.title('$B_0$ and axis shape is irrelevant \n' + r'$\hat{\omega}_{T}$' + ' = {} , '.format(omt) + r'$\bar{\eta}$' + ' = {}'.format(eta), size = 18)

# plt.axis([0, 10, 0, 10])
plt.text(2e-6, 5e-2, r'$\hat{\omega}_{n}$' + ' = {} , '.format(omn) + r'$\bar{\eta}$' + ' = {}'.format(eta))
#plt.text(5e-5, 2e-5

#plt.text(2e-6, 2e-2

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$r$', size = 28)
plt.ylabel(r'$\hat{A}$', size = 28)
plt.legend(loc = 'best', fontsize = 20)
plt.grid(alpha = 0.5)
plt.savefig('THESIS_OMT_r_ae_nor_eta_{}.png'.format(-eta), bbox_inches='tight', dpi = 300)
plt.show()
