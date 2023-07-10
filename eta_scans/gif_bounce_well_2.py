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
dln_n_dpsi = -5
dln_T_dpsi = 0

# defining physics variables
# mutiple etas

eta_N = 20
eta_arr = np.linspace(-3, -0.1, eta_N)

# 1 or 3 gradients
omn = dln_n_dpsi
omT = dln_T_dpsi

ae_total_arr = np.empty([eta_N, 1]) # this could be 2 or 3 ... if you want more gradients

ae_per_lam_arr = np.empty([lam_res - 2, eta_N, 1]) # this could be 2 or 3 ... if you want more gradients

ae_per_lam_nor_arr = np.empty([lam_res - 2, eta_N, 1])

w_alpha_arr = np.empty([lam_res - 2, eta_N, 1])

# lam arr

for idx, eta in enumerate(eta_arr):

    stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
    w_alpha_arr[:, idx, 0], ae_per_lam_arr[:, idx, 0], ae_total_arr[idx, 0] = ae_nae(stel, r, lam_res, Delta_psiA, omn, omT, plot = False)[1:]
    ae_per_lam_nor_arr[:, idx, 0] = ae_per_lam_arr[:, idx, 0]/ae_total_arr[idx, 0]

# defining figure variables

fig = plt.figure(figsize = (12, 9))

plt.tight_layout()

grid = plt.GridSpec(3, 3, wspace =0.1, hspace = 0.3)

ax = plt.subplot(grid[2, :])

ax1 = plt.subplot(grid[0:2, :])

ax2=ax1.twinx()

#### normalizing colorbar

ae_per_lam_min = np.min(ae_per_lam_arr.flatten())

ae_per_lam_max = 250 #np.max(ae_per_lam_arr.flatten())

cmap = plt.get_cmap('plasma', 200)
norm = mpl.colors.Normalize(vmin=ae_per_lam_min, vmax=ae_per_lam_max)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

fig.suptitle('Analytical, r = {} [m], B0 = {} [T], \n'.format(r, B0) + \
 r'$\Delta{\psi}_{A}$' + ' = {:.2f}'.format(Delta_psiA) + \
', omT = {}, omn = {}'.format(dln_T_dpsi, dln_n_dpsi) , size = 16)

cbar = fig.colorbar(sm, ax=ax1, pad = 0.30, location = 'left')#, label = r'$\hat{A}_{\lambda}/\hat{A}$')
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_title(label = r'$\hat{A}_{\lambda}$', fontsize=16)

pos = ax1.get_position()
new_pos = [pos.x0-0.12, pos.y0, pos.width, pos.height]
ax1.set_position(new_pos)

# function for doing animation

def update_linechart(i):

  for j in range(i+1):

      ax.clear()
      ax.cla()

      ax1.clear()
      #cax.cla()
      ax1.cla()

      #caxx.cla()
      ax2.cla()
      ax2.clear()

      #ax1.clf()


      # defining ae_per_lamb

      eta = eta_arr[j]

      stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)

      # things for plotting the well
      res = 5000

      theta = np.linspace(-np.pi, np.pi, res)

      N = stel.iota - stel.iotaN

      phi = theta/(stel.iotaN + N)

      vartheta = theta - N*phi

      lamb_min = lamb_min = 1/(1 - r*eta)

      lamb_max = lamb_max = 1/(1 + r*eta)

      lam_arr = np.linspace(lamb_min, lamb_max, lam_res)[1:-1]

      vartheta_ae = 2*np.arcsin(np.sqrt((((1 - lam_arr*(1 + r*eta))/(-r*eta*lam_arr)))/2))

      magB = stel.B_mag(r=r, theta=theta, phi=phi)

      norm = plt.Normalize()
      color_arr = np.concatenate((ae_per_lam_arr[:, j, 0], np.array([ae_per_lam_min, ae_per_lam_max])))
      #print(color_arr.shape)
      colors = plt.cm.plasma(norm(color_arr))

      # cmap = plt.get_cmap('plasma', 200)
      # norm = mpl.colors.Normalize(vmin=np.min(ae_per_lam_nor_arr[:, j, 0]), vmax=np.max(ae_per_lam_nor_arr[:, j, 0]))
      # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
      # sm.set_array([])

      for idx, var_theta in enumerate(vartheta_ae):

          B_vartheta = np.interp(var_theta, vartheta, magB)
          ax1.plot([-var_theta, var_theta], [B_vartheta, B_vartheta], c = colors[idx], linewidth = 0.9)

      ax1.plot(vartheta, magB, c = 'black')
      ax1.set_ylabel('$|B|$', size = 18)
      ax1.set_xlabel(r'$\vartheta$', size = 18)


      # vmin = np.min(ae_per_lam_nor_arr[:, j, 0])  # get minimum of nth channel
      # vmax = vmax=np.max(ae_per_lam_nor_arr[:, j, 0])  # get maximum of nth channel
      # cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
      # cbar.draw_all()


      ax2.plot(vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 0.9, alpha = 0.7)
      ax2.plot(-vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 0.9, alpha = 0.7)

      ax2.set_ylabel(r'$\hat{\omega}_{\alpha}$', color = 'r', size = 18)
      ax2.spines['right'].set_color('r')
      ax2.tick_params(colors='red', which='major')


      ax.plot(eta_arr, ae_total_arr[:,0])
      ax.scatter(eta_arr[j], ae_total_arr[j,0], label = '$\eta$ = {:.3f}, AE total = {:.3f}'.format(eta_arr[j], ae_total_arr[j,0]))
      ax.set_yscale('log')
      ax.set_xlabel(r'$\bar{\eta}$', size = 18)
      ax.set_ylabel(r'$\hat{A}$', size = 18)
      ax.legend(fontsize = 12)

      # bounce well plot

num_frames = 5 #len(eta_arr)
anim = animation.FuncAnimation(fig, update_linechart, frames = num_frames, interval = 150)
anim.save('animate_well{}.gif'.format(dln_n_dpsi))
