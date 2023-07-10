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

############### QA ###################

# optimal QA simple model nfp 2

eta = -1.0961558840654466      # parameter

nfp = 2         # number of field periods

rc=[1, 0.23587939698492463 ]   # rc -> the cosine components of the axis radial component
zs=[0, 0.23587939698492463 ]  # zs -> the sine components of the axis vertical component


# optimal QA simple model nfp 3

eta = -1.551538720240993       # parameter

nfp = 3         # number of field periods

rc=[1, 0.15864321608040202]   # rc -> the cosine components of the axis radial component
zs=[0, 0.15864321608040202]  # zs -> the sine components of the axis vertical component


# optimal QA simple model nfp 4

eta = -2.0213973233891216      # parameter

nfp = 4         # number of field periods

rc=[1, 0.1222110552763819 ]   # rc -> the cosine components of the axis radial component
zs=[0, 0.1222110552763819 ]  # zs -> the sine components of the axis vertical component

# optimal QA simple model nfp 5

eta = -1.551538720240993      # parameter

nfp = 5        # number of field periods

rc=[1, 0.15864321608040202]   # rc -> the cosine components of the axis radial component
zs=[0, 0.15864321608040202]  # zs -> the sine components of the axis vertical component


############### QH ###################

# optimal QH simple model nfp 2

eta = -1.0961558840654466      # parameter

nfp = 2         # number of field periods

rc=[1, 0.23587939698492463 ]   # rc -> the cosine components of the axis radial component
zs=[0, 0.23587939698492463 ]  # zs -> the sine components of the axis vertical component


# optimal QH simple model nfp 3

eta = -1.551538720240993       # parameter

nfp = 3         # number of field periods

rc=[1, 0.15864321608040202]   # rc -> the cosine components of the axis radial component
zs=[0, 0.15864321608040202]  # zs -> the sine components of the axis vertical component


# optimal QH simple model nfp 4

eta = -2.0213973233891216      # parameter

nfp = 4         # number of field periods

rc=[1, 0.1222110552763819 ]   # rc -> the cosine components of the axis radial component
zs=[0, 0.1222110552763819 ]  # zs -> the sine components of the axis vertical component

# optimal QH simple model nfp 5

eta = -2.49283883215899      # parameter

nfp = 5        # number of field periods

rc=[1, 0.10180904522613064]   # rc -> the cosine components of the axis radial component
zs=[0, 0.10180904522613064]  # zs -> the sine components of the axis vertical component

####

nfp = 3
B0 = 1
rc = [1, 0.045]
zs = [0, 0.045]
eta = -0.1

##########################################################################################

# these quantities are provided in [m]

# this is the configuration of r1 section 5.1 from
# (not sure) Landreman, Sengupta, and Plunk, Journal of Plasma Physics 85, 905850103 (2019).

# stel_1 = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=-2.5, B0=B0)
# stel_2 = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=-0.9, B0=B0)
# stel_3 = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=-0.2, B0=B0)

# to calculate things the lam_res needs to be provided, just that?
lam_res = 10000

# distance from the magnetic axis to be used
r = 0.005

a_r = 1

# stel_1.plot_boundary(r = r)
# stel_2.plot_boundary(r = r)
# stel_3.plot_boundary(r = r)

# normalization variable for AE
Delta_r = 1

# gradients for diamagnetic frequency
dln_n_dpsi = -5
dln_T_dpsi = 0


################# PLOT 1 ######################

# scans over gradients n
omn_N = 10
omn_arr = np.linspace(-5, 5, omn_N)

# scans over gradients T
omT_N = 10
omT_arr = np.linspace(-5, 5, omT_N)

#scans over etabar
eta_N = 100
eta_arr = np.linspace(-3, -0.1, eta_N)

# array to get the ae total
# ae_total_arr = np.empty_like(eta_arr)
ae_total_arr_n = np.empty([omn_N, eta_N])

ae_total_arr_T = np.empty([omT_N, eta_N])

# optimize eta arrays

eta_min_omn_arr = np.empty(omn_N)

eta_min_omT_arr = np.empty(omT_N)


# iota arrays

iota_omn_arr = np.empty([omn_N, eta_N])
iota_omT_arr = np.empty([omT_N, eta_N])

iota_opt_omn_arr = np.empty(omn_N)
iota_opt_omT_arr = np.empty(omT_N)

# stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
# stel.plot_boundary(r = r)

# print("alpha tilde: ")
# print(stel.iota - stel.iotaN)

nfp = 3
B0 = 1
rc = [1, 0.045]
zs = [0, 0.045]
eta = -100


eta_arr = np.geomspace(-700, -0.01, 100)
ae_total_arr = np.empty_like(eta_arr)
ae_nor_total_arr = np.empty_like(eta_arr)

# print(eta_arr)


for idx, eta in enumerate(eta_arr):
    
    stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
    ae_total_arr[idx], ae_nor_total_arr[idx] = ae_nor_nae(stel, r, lam_res, Delta_r, a_r, dln_n_dpsi, dln_T_dpsi, plot = False)[3:]
    #print(idx)


print(ae_total_arr)

plt.plot(eta_arr, ae_nor_total_arr)
plt.plot(eta_arr, ae_total_arr)
plt.yscale('log')
plt.xlabel(r'$\bar{\eta}$', size = 16)
plt.ylabel(r'$\frac{A}{E_t}$', size = 16)
plt.show()


# getting the data



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

