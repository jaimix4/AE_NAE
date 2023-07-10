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
lam_res = 20000

# distance from the magnetic axis to be used

r = 0.001

a_r = 1

# normalization variable for AE

Delta_r = 1

# gradients for diamagnetic frequency
omn = -5
omt = 0

# gradient array
N = 10
#omn_arr = np.linspace(-20, -1, N)
omn_arr = -1*np.geomspace(1e-3, 1, N)
idx_eta_dagger_arr = np.zeros(N).astype(int)

# arrays for plotting
N_eta = 100
eta_arr = np.geomspace(-800, -0.001, N_eta)

# empty array with shape N by N_eta
ae_total_arr = np.empty((N, N_eta))
ae_nor_total_arr = np.empty((N, N_eta))

eta_arr_4plot = -1*eta_arr

plt.figure(figsize = (12,7))


for idx_omn, omn in enumerate(omn_arr):
    for idx, eta in enumerate(eta_arr):

        stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
        ae_total_arr[idx_omn, idx], ae_nor_total_arr[idx_omn, idx] = ae_nor_nae(stel, r, lam_res, Delta_r, a_r, omn, omt, plot = False)[3:]
  
    idx_eta_dagger_arr[idx_omn] = np.argmax(ae_nor_total_arr[idx_omn, :])

    plt.plot(eta_arr_4plot, ae_nor_total_arr[idx_omn, :], label='omn = {:.5f} ,'.format(omn) + r'$\bar{\eta}_{\dag}$' + ' = {:.3f}'.format(eta_arr[idx_eta_dagger_arr[idx_omn]]))
    plt.scatter(eta_arr_4plot[idx_eta_dagger_arr[idx_omn]], ae_nor_total_arr[idx_omn, idx_eta_dagger_arr[idx_omn]])

plt.title('B0 and axis shape is irrelevant \n r = {}, omt = {}'.format(r, omt), size = 20)
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$-\bar{\eta}$', size = 20)
plt.ylabel(r'$\hat{A} = A/E_t$', size = 20)
plt.legend(loc = 'best', fontsize = 12)
plt.grid(alpha = 0.5)
plt.savefig('ae_nor_eta_dagger_r_{}_omn_new.png'.format(r), bbox_inches='tight', dpi = 300)
plt.show()



"""

for omn_idxN, dln_n_dpsi in enumerate(omn_arr):

    for idx, eta in enumerate(eta_arr):

        stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
        ae_total_arr_n[omn_idxN, idx] = ae_nor_nae(stel, r, lam_res, Delta_r, a_r, dln_n_dpsi, 0, plot = False)[4]
        iota_omn_arr[omn_idxN, idx] = stel.iotaN

    idx_min = np.argmin(ae_total_arr_n[omn_idxN, :])

    eta_min_omn_arr[omn_idxN] = eta_arr[idx_min]
    iota_opt_omn_arr[omn_idxN] = iota_omn_arr[omn_idxN, idx_min]

for omT_idxN, dln_T_dpsi in enumerate(omT_arr):

    for idx, eta in enumerate(eta_arr):

        stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
        ae_total_arr_T[omT_idxN, idx] = ae_nor_nae(stel, r, lam_res, Delta_r, a_r, 0, dln_T_dpsi, plot = False)[4]
        iota_omT_arr[omT_idxN, idx] = stel.iotaN

    idx_min = np.argmin(ae_total_arr_T[omT_idxN, :])

    eta_min_omT_arr[omT_idxN] = eta_arr[idx_min]
    iota_opt_omT_arr[omT_idxN] = iota_omT_arr[omT_idxN, idx_min]

# getting max iota

iota_max_idx = np.argmax(iota_omn_arr[0,:])
iota_max = iota_omn_arr[0, iota_max_idx]

# print(iota_max)
# print(eta_arr[iota_max_idx])

# Plotting
#### PLOT 1 ####

fig,ax = plt.subplots(figsize=(12,7))

ax2=ax.twinx()

ax2.plot(eta_arr, iota_omn_arr[0, :], 'k', zorder = -1)
ax2.scatter(eta_arr[iota_max_idx], iota_max, color='k', zorder = -1)
ax2.scatter(eta_arr[iota_max_idx], iota_max, alpha = 0.0, zorder = -1)
ax2.set_ylabel(r'$\iota_N$', size = 18)

ax.plot(eta_arr, ae_total_arr_n[0, :], alpha = 0.0, label = "omn, eta_min")
ax.scatter(eta_arr[0], ae_total_arr_n[0, 0], alpha = 0.0)

for omn_idxN, dln_n_dpsi in enumerate(omn_arr):

    if dln_n_dpsi < 0:
        line = '-.'
    else:
        line = '-'

    idx_min = np.argmin(ae_total_arr_n[omn_idxN, :])
    ax.scatter(eta_arr[idx_min], ae_total_arr_n[omn_idxN, idx_min])

    ax.plot(eta_arr, ae_total_arr_n[omn_idxN, :], line, label = "{:.2f}, {:.3f}".format(dln_n_dpsi, eta_arr[idx_min]))

    ax2.scatter(eta_arr[idx_min], iota_omn_arr[omn_idxN, idx_min])



ax.set_title('Shape: rc=' + str(rc) + ', zs = ' + str(zs) \
            + '\n r = {}, B0 = {}, nfp = {}, omT = 0'.format(r, B0, nfp), size = 16) \

ax.set_yscale('log')
ax.set_xlabel(r'$\bar{\eta}$', size = 18)
ax.set_ylabel(r'$\hat{A}$', size = 18)
ax.legend(loc = 2, fontsize = 12).set_zorder(10)
ax.grid(alpha = 0.3, linewidth = 0.5)
plt.tight_layout()
plt.savefig('nor_qh_eta{}_scan_fig_1.png'.format(-eta), bbox_inches='tight', dpi = 300)
plt.show()

### PLOT 2 ###

# Plotting
fig,ax = plt.subplots(figsize=(12,7))

ax2=ax.twinx()

ax2.plot(eta_arr, iota_omn_arr[0, :], 'k', zorder = -1)
ax2.scatter(eta_arr[iota_max_idx], iota_max, color='k', zorder = -1)
ax2.scatter(eta_arr[iota_max_idx], iota_max, alpha = 0.0, zorder = -1)
ax2.set_ylabel(r'$\iota_N$', size = 18)

ax.plot(eta_arr, ae_total_arr_n[0, :], alpha = 0.0, label = "omT, eta_min")
ax.scatter(eta_arr[0], ae_total_arr_n[0, 0], alpha = 0.0)

for omT_idxN, dln_T_dpsi in enumerate(omT_arr):

    if dln_T_dpsi < 0:
        line = '-.'
    else:
        line = '-'

    idx_min = np.argmin(ae_total_arr_T[omT_idxN, :])
    ax.scatter(eta_arr[idx_min], ae_total_arr_T[omT_idxN, idx_min])

    ax.plot(eta_arr, ae_total_arr_T[omT_idxN, :], line, label = "{:.2f}, {:.3f}".format(dln_T_dpsi, eta_arr[idx_min]))

    ax2.scatter(eta_arr[idx_min], iota_omT_arr[omT_idxN, idx_min])

ax.set_title('Shape: rc=' + str(rc) + ', zs = ' + str(zs) \
            + '\n r = {}, B0 = {}, nfp = {}, omn = 0'.format(r, B0, nfp), size = 16) \

ax.set_yscale('log')
ax.set_xlabel(r'$\bar{\eta}$', size = 18)
ax.set_ylabel(r'$\hat{A}$', size = 18)
ax.legend(loc = 2, fontsize = 12).set_zorder(10)
ax.grid(alpha = 0.3, linewidth = 0.5)
plt.tight_layout()
plt.savefig('nor_qh_eta{}_scan_fig_2.png'.format(-eta), bbox_inches='tight', dpi = 300)
plt.show()

####### computing optimize etabar



#### PLOT 3 ####

fig,ax = plt.subplots(figsize=(7,7))

ax.plot(omn_arr, eta_min_omn_arr, '-*r', label = "eta omn, omT = 0")
ax.plot(omT_arr, eta_min_omT_arr, '-*b', label = "eta omT, omn = 0")

ax2=ax.twinx()

ax2.plot(omn_arr, iota_opt_omn_arr, '-.*r', alpha = 0.25, label = "iota omn, omT = 0")
ax2.plot(omT_arr, iota_opt_omT_arr, '-.*b', alpha = 0.25, label = "iota omT, omn = 0")
ax2.plot(omT_arr, iota_max*np.ones_like(omT_arr), '-k', alpha = 0.7, label = "iota max")

ax2.set_ylabel(r'$\iota_N$' + ' (min AE)', size = 18)


ax.set_title('Shape: rc=' + str(rc) + ', zs = ' + str(zs) \
            + '\n r = {}, B0 = {}, nfp = {}'.format(r, B0, nfp, dln_T_dpsi), size = 16) \

ax.set_xlabel('gradient', size = 18)
ax.set_ylabel(r'$\bar{\eta}$' + ' (min AE)', size = 18)
ax.legend(fontsize = 12)
ax2.legend(fontsize = 12)
ax.grid(alpha = 0.3, linewidth = 0.5)
plt.tight_layout()
plt.savefig('nor_qh_eta{}_scan_fig_3.png'.format(-eta), bbox_inches='tight', dpi = 300)
plt.show()

"""