import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qsc import Qsc
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_nor_func_pyqsc import ae_nor_nae
from opt_eta_star import set_eta_star
import time
from scipy.interpolate import interp2d, interp1d

# A configuration is defined with B0, etabar, field periods, Fourier components and eta bar

B0 = 1         # [T] strength of the magnetic field on axis

eta = -0.9      # parameter

nfp = 3         # number of field periods

rc=[1, 0.045]   # rc -> the cosine components of the axis radial component
zs=[0, 0.045]  # zs -> the sine components of the axis vertical component

# these quantities are provided in [m]

# this is the configuration of r1 section 5.1 from
# (not sure) Landreman, Sengupta, and Plunk, Journal of Plasma Physics 85, 905850103 (2019).

# stel_1 = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=-2.5, B0=B0)
# stel_2 = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=-0.9, B0=B0)
# stel_3 = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=-0.2, B0=B0)

# to calculate things the lam_res needs to be provided, just that?
lam_res = 5000
Delta_r = 1
a_r = 1

# distance from the magnetic axis to be used
r = 0.01

# stel_1.plot_boundary(r = r)
# stel_2.plot_boundary(r = r)
# stel_3.plot_boundary(r = r)

# normalization variable for AE
Delta_psiA = 1

# gradients for diamagnetic frequency
# dln_n_dpsi = -5
# dln_T_dpsi = 0

# defining physics variables
# mutiple etas

eta_N = 30
eta_arr = np.linspace(-2.5, -0.05, eta_N)

# 1 or 3 gradients
omn = 0.5
omt = 0.5

ae_total_arr = np.empty([eta_N, 1]) # this could be 2 or 3 ... if you want more gradients

ae_per_lam_arr = np.empty([lam_res - 2, eta_N, 1]) # this could be 2 or 3 ... if you want more gradients

ae_per_lam_nor_arr = np.empty([lam_res - 2, eta_N, 1])

w_alpha_arr = np.empty([lam_res - 2, eta_N, 1])

# lam arr

for idx, eta in enumerate(eta_arr):

    #stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
    #w_alpha_arr[:, idx, 0], ae_per_lam_arr[:, idx, 0], ae_total_arr[idx, 0] = ae_nae(stel, r, lam_res, Delta_psiA, omn, omT, plot = False)[1:]
    #return tau_nor, w_alpha_nor, ae_per_lam, self.ae_total, self.ae_nor_total
    w_alpha_arr[:, idx, 0], ae_per_lam_arr[:, idx, 0],_ , ae_total_arr[idx, 0] = \
    ae_nor_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0), r, lam_res, Delta_r, a_r, omn, omt, plot = False)[1:]
    
    ae_per_lam_nor_arr[:, idx, 0] = ae_per_lam_arr[:, idx, 0]/np.max(ae_per_lam_arr[:, idx, 0])


# find the index of the minimum ae_total_arr
idx_min = np.argmin(ae_total_arr)
print("Min eta = {}".format( eta_arr[idx_min] ))


# defining figure variables

fig = plt.figure(figsize = (9, 7))

plt.tight_layout()

grid = plt.GridSpec(3, 3, wspace = 0.5, hspace = 0.5, width_ratios=[1.7, 1, 1], height_ratios=[1, 1, 1.7])

ax = plt.subplot(grid[2, 1:])

ax1 = plt.subplot(grid[0:2, :])

ax2=ax1.twinx()

ax3 = plt.subplot(grid[2, 0])

# grid.tight_layout(fig)
grid.update(top=0.95)

#### normalizing colorbar

ae_per_lam_min = np.min(ae_per_lam_nor_arr.flatten())

ae_per_lam_max = np.max(ae_per_lam_nor_arr.flatten())

cmap = plt.get_cmap('plasma', 200)
norm = mpl.colors.Normalize(vmin=ae_per_lam_min, vmax=ae_per_lam_max)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

#  r'$\Delta{\psi}_{A}$' + ' = {:.2f}'.format(Delta_psiA) + \
# fig.suptitle('Analytical, r = {} [m], B0 = {} [T], \n'.format(r, B0) + \
# 'omt = {}, omn = {}'.format(omt, omn) , size = 16)

cbar = fig.colorbar(sm, ax=ax1, pad = 0.30, location = 'left')#, label = r'$\hat{A}_{\lambda}/\hat{A}$')
cbar.ax.tick_params(labelsize=16)
# cbar.ax.set_title(label = r'$\hat{A}_{\lambda}/\hat{A}_{\rm max. \lambda}$', fontsize=16)

pos = ax1.get_position()
new_pos = [pos.x0-0.12, pos.y0, pos.width, pos.height]
ax1.set_position(new_pos)

# function for boundaries
#  
def plot_boundary(self, r=0.1, ntheta=80, nphi=150, ntheta_fourier=20, nsections=8,
         fieldlines=False, savefig=None, colormap=None, azim_default=None,
         show=True, **kwargs):

    x_2D_plot, y_2D_plot, z_2D_plot, R_2D_plot = self.get_boundary(r=r, ntheta=ntheta, nphi=nphi, ntheta_fourier=ntheta_fourier)
    phi = np.linspace(0, 2 * np.pi, nphi)  # Endpoint = true and no nfp factor, because this is what is used in get_boundary()
    R_2D_spline = interp1d(phi, R_2D_plot, axis=1)
    z_2D_spline = interp1d(phi, z_2D_plot, axis=1)

    ## Poloidal plot
    phi1dplot_RZ = np.linspace(0, 2 * np.pi / self.nfp, nsections, endpoint=False)
    # fig = plt.figure(figsize=(6, 6), dpi=80)
    # ax  = plt.gca()
    for i, phi in enumerate(phi1dplot_RZ):
        phinorm = phi * self.nfp / (2 * np.pi)
        if phinorm == 0:
            label = r'$\phi$=0'
        elif phinorm == 0.25:
            label = r'$\phi={\pi}/$' + str(2 * self.nfp)
        elif phinorm == 0.5:
            label = r'$\phi=\pi/$' + str(self.nfp)
        elif phinorm == 0.75:
            label = r'$\phi={3\pi}/$' + str(2 * self.nfp)
        else:
            label = '_nolegend_'
        color = next(ax._get_lines.prop_cycler)['color']
        # Plot location of the axis
        ax3.plot(self.R0_func(phi), self.Z0_func(phi), marker="x", linewidth=2, label=label, color=color)
        # Plot poloidal cross-section
        ax3.plot(R_2D_spline(phi), z_2D_spline(phi), color=color)
    # plt.xlabel('R (meters)')
    # plt.ylabel('Z (meters)')
    # plt.legend()
    #plt.tight_layout()
    ax3.set_aspect('equal')
    # plt.show()



# function for doing animation

def update_linechart(i):

  for j in range(i+1):

      ax.clear()
      ax.cla()

      ax1.clear()
      #cax.cla()
      ax1.cla()
      #ax2=ax1.twinx()
      #caxx.cla()
      ax2.cla()
      ax2.clear()

      ax3.clear()
      ax3.cla()

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
      ax1.tick_params(axis='both', which='major', labelsize=16)
    #   ax1.set_ylabel('$|B|$', size = 18)
    #   ax1.set_xlabel(r'$\vartheta$', size = 18)

      
      # vmin = np.min(ae_per_lam_nor_arr[:, j, 0])  # get minimum of nth channel
      # vmax = vmax=np.max(ae_per_lam_nor_arr[:, j, 0])  # get maximum of nth channel
      # cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
      # cbar.draw_all()


      ax2.plot(vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 3, alpha = 0.8)
      ax2.plot(-vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 3, alpha = 0.8, label = r'$\hat{\omega}_{\alpha}$')

      # find where walpha is close to zero


      ax2.plot(vartheta_ae, 0*w_alpha_arr[:, j, 0], '-.b', linewidth = 3, alpha = 0.8)
      ax2.plot(-vartheta_ae, 0*w_alpha_arr[:, j, 0], '-.b', linewidth = 3, alpha = 0.8, label = r'$\hat{\omega}_{\psi}$')

      #ax2.set_ylabel(r'$\hat{\omega}_{\alpha}$', color = 'r', size = 18)
      ax2.spines['right'].set_color('r')
      ax2.tick_params(colors='red', which='major')
      # ax2.legend(fontsize = 20)
      # adjust size of tick labels
      ax2.tick_params(axis='both', which='major', labelsize=16)


      ax.plot(eta_arr, ae_total_arr[:,0], 'b', linewidth = 3) 
      # how to adjust the size and mark of the scatter points
      ax.scatter(eta_arr[j], ae_total_arr[j,0], s=55 , c='b', label = r'$\bar{\eta}$' + ' = {:.2f}, '.format(eta_arr[j]) + r'$\hat{A}$' + ' = {:.3f}'.format(ae_total_arr[j,0]))
      #ax.set_ylim(np.min(ae_total_arr[:,0])-0.001, 0.1)

      ax.set_yscale('log')
      ax.grid(alpha = 0.5)
    #   ax.set_xlabel(r'$\bar{\eta}$', size = 18)
    #   ax.set_ylabel(r'$\hat{A}$', size = 18)
      ax.legend(fontsize = 17)
      ax.tick_params(axis='both', which='major', labelsize=16)

      # bounce well plot

      # cross section plots 
      ax3.tick_params(axis='both', which='major', labelsize=16)
      ax3.axis('off')
      # limits of axes
    #   ax3.set_ylim(-0.8, 0.8)
    #   ax3.set_xlim(0.8, 1.5)
      plot_boundary(stel, r = 0.03 )

    #   fig.tight_layout()


num_frames = len(eta_arr)
anim = animation.FuncAnimation(fig, update_linechart, frames = num_frames, interval = 150)
anim.save('animate_well{}_changed.gif'.format(omn))
