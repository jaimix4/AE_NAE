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

B0 = 2         # [T] strength of the magnetic field on axis

eta = -0.9      # parameter

nfp = 3         # number of field periods

rc=[1, 0.045]   # rc -> the cosine components of the axis radial component
zs=[0, -0.045]  # zs -> the sine components of the axis vertical component

# these quantities are provided in [m]

# this is the configuration of r1 section 5.1 from
# (not sure) Landreman, Sengupta, and Plunk, Journal of Plasma Physics 85, 905850103 (2019).

# stel_1 = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=-2.5, B0=B0)
# stel_2 = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=-0.9, B0=B0)
# stel_3 = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=-0.2, B0=B0)

# to calculate things the lam_res needs to be provided, just that?
lam_res = 5000

# distance from the magnetic axis to be used
r = 0.05

# stel_1.plot_boundary(r = r)
# stel_2.plot_boundary(r = r)
# stel_3.plot_boundary(r = r)

# normalization variable for AE
Delta_psiA = 1

# gradients for diamagnetic frequency
dln_n_dpsi = 5
dln_T_dpsi = 0


#omn_arr = np.array([-1])

################# PLOT 1 ######################
#scans over etabar
# scans over gradients

"""

omn_arr = np.linspace(-5, 5, 10)

eta_arr = np.linspace(-3, -0.1, 10)
ae_total_arr = np.empty_like(eta_arr)

fig,ax = plt.subplots(figsize=(12,7))

for dln_n_dpsi in omn_arr:

    ae_total_arr = np.empty_like(eta_arr)

    for idx, eta in enumerate(eta_arr):

        stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
        ae_total_arr[idx] = ae_nae(stel, r, lam_res, Delta_psiA, dln_n_dpsi, dln_T_dpsi, plot = False)[3]

    if dln_n_dpsi < 0:
        line = '-.'
    else:
        line = '-'

    ax.plot(eta_arr, ae_total_arr, line, label = "omn = {:.2f}".format(dln_n_dpsi))

ax.set_title('Shape: rc=' + str(rc) + ', zs = ' + str(zs) \
            + '\n r = {}, B0 = {}, nfp = {}, omT = {} '.format(r, B0, nfp, dln_T_dpsi)) \

ax.set_yscale('log')
ax.set_xlabel(r'$\bar{\eta}$')
ax.set_ylabel(r'$\hat{A}$')
ax.legend()

"""

#plt.show()


################# PLOT 2 ######################
# scans over etabar and iota
# scans over gradients

#ax2.set_ylabel(r'$\iota$', color = 'r')
# ax2.spines['right'].set_color('r')
# ax2.tick_params(colors='red', which='major')


################# PLOT 3 ######################
# scans over etabar and iota
# scans over gradients
# GIF

# defining figure variables

fig = plt.figure(figsize = (10, 13))
grid = plt.GridSpec(3, 3, wspace =0.3, hspace = 0.3)

ax = plt.subplot(grid[2, :])

ax1 = plt.subplot(grid[0:2, :])

divider = make_axes_locatable(ax1)

cax = divider.append_axes("left", size="5%", pad=0.9)

ax2=ax1.twinx()

divider = make_axes_locatable(ax2)

caxx = divider.append_axes("left", size="5%", pad=0.9)

# defining physics variables

omn_arr = np.linspace(-5, 5, 10)

eta_arr = np.linspace(-3, -0.1, 10)

ae_total_arr = np.empty_like(eta_arr)

omn_arr = np.linspace(-5, 5, 10)

eta_arr = np.linspace(-3, -0.1, 10)
ae_total_arr = np.empty_like(eta_arr)


for idx, eta in enumerate(eta_arr):

    stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
    ae_total_arr[idx] = ae_nae(stel, r, lam_res, Delta_psiA, dln_n_dpsi, dln_T_dpsi, plot = False)[3]

def update_linechart(i):

  for j in range(i):

      ax.clear()
      ax.cla()

      ax1.clear()
      cax.cla()
      ax1.cla()

      caxx.cla()
      ax2.cla()
      ax2.clear()

      # plot down

      ax.plot(eta_arr, ae_total_arr)
      ax.scatter(eta_arr[j], ae_total_arr[j])
      ax.set_yscale('log')

      # bounce well plot

      eta = eta_arr[j]
      stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
      ae_nae(stel, r, lam_res, Delta_psiA, dln_n_dpsi, dln_T_dpsi, ax = ax1, ax2 = ax2, cax = cax, plot = False)


num_frames = len(eta_arr)
anim = animation.FuncAnimation(fig, update_linechart, frames = num_frames, interval = 200)
anim.save('linechart.gif')

#plt.show()
