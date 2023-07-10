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

from   matplotlib        import rc
# add latex fonts to plots
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 22})
rc('text', usetex=True)

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

eta_arr[17] = -0.82

# 1 or 3 gradients
omn = 1
omt = 1

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

fig = plt.figure(figsize = (24.5/1.5, 20/1.5))

# plt.tight_layout()

grid = plt.GridSpec(4, 4, wspace = 0.8, hspace = 0.3, width_ratios = [0.05, 1, 0.5, 0.3])#, width_ratios=[1.7, 1, 1],  os=[1, 1, 1.7])

ax1 = plt.subplot(grid[0, 1])
ax1_ae = plt.subplot(grid[0, 2])
ax1_cross = plt.subplot(grid[0, 3])


ax2 = plt.subplot(grid[1, 1])
ax2_ae = plt.subplot(grid[1, 2])
ax2_cross = plt.subplot(grid[1, 3])

ax3 = plt.subplot(grid[2, 1])
ax3_ae = plt.subplot(grid[2, 2])
ax3_cross = plt.subplot(grid[2, 3])

ax4 = plt.subplot(grid[3, 1])
ax4_ae = plt.subplot(grid[3, 2])
ax4_cross = plt.subplot(grid[3, 3])


########## normalising colo bar ############

ae_per_lam_min = np.min(ae_per_lam_nor_arr.flatten())

ae_per_lam_max = np.max(ae_per_lam_nor_arr.flatten())

cmap = plt.get_cmap('plasma', 200)
norm = mpl.colors.Normalize(vmin=ae_per_lam_min, vmax=ae_per_lam_max)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

#  r'$\Delta{\psi}_{A}$' + ' = {:.2f}'.format(Delta_psiA) + \
# fig.suptitle('Analytical, r = {} [m], B0 = {} [T], \n'.format(r, B0) + \
# 'omt = {}, omn = {}'.format(omt, omn) , size = 16)

# put color bar on top across all subplots

# Create a horizontal color bar outside the grid
# cax = fig.add_axes([0.234, 0.902, 0.215, 0.018]) # [left, bottom, width, height]
# cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')

cax = fig.add_subplot(grid[:, 0])  # Colorbar subplot for plots 1 and 2
cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
cbar.ax.locator_params(nbins=8)

# Adjust the size of color bar ticks
cbar.ax.tick_params(labelsize=20, direction='out')
cbar.ax.xaxis.set_label_position('top')

# label for color bar to the right
cbar.ax.set_xlabel(r'$\hat{A}_{\lambda}/\hat{A}_{\lambda^{\rm max.}}$', labelpad=20, fontsize = 24)




# Set the label on the right side of the color bar
# cbar.ax.xaxis.set_label_position('top')

# cbar.ax.set_xlabel(r'$\hat{A}_{\lambda}/\hat{A}$', labelpad=10, fontsize = 24)

# divider = make_axes_locatable(ax1)
# cax = divider.append_axes("top", size="10%", pad=0.2)
# plt.colorbar(sm, cax=cax, orientation="horizontal", label = r'$\hat{A}_{\lambda}/\hat{A}$')

# cbar = fig.colorbar(sm, ax=ax1, pad = 0.30, location = 'top')#, label = r'$\hat{A}_{\lambda}/\hat{A}$')
# cbar.ax.tick_params(labelsize=16)

########### first plot ############


j = 0

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
ax1.tick_params(axis='both', which='major', labelsize=20)
#ax1.set_ylabel('$|B|$', size = 28, pad = 10, rotation = 0)
#ax1.set_title( r'$\hat{A}_{\lambda}/\hat{A}_{\rm max.}$', loc = 'left', size = 24, pad = 5)

ax1.set_title( r'$\left|B\right|$', loc = 'left', size = 28, pad = 20)
#   ax1.set_xlabel(r'$\vartheta$', size = 18)


# vmin = np.min(ae_per_lam_nor_arr[:, j, 0])  # get minimum of nth channel
# vmax = vmax=np.max(ae_per_lam_nor_arr[:, j, 0])  # get maximum of nth channel
# cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
# cbar.draw_all()

ax1_w = ax1.twinx()

ax1_w.plot(vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 3, alpha = 0.8)
ax1_w.plot(-vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 3, alpha = 0.8, label = r'$\hat{\omega}_{\alpha}$')

# find where walpha is close to zero


ax1_w.plot(vartheta_ae, 0*w_alpha_arr[:, j, 0], '-.b', linewidth = 3, alpha = 0.8)
ax1_w.plot(-vartheta_ae, 0*w_alpha_arr[:, j, 0], '-.b', linewidth = 3, alpha = 0.8, label = r'$\hat{\omega}_{\psi}$')

#ax1_w.set_ylabel(r'$\hat{\omega}_{\alpha}$', color = 'r', size = 24, loc='bottom')
ax1_w.spines['right'].set_color('r')
ax1_w.tick_params(colors='red', which='major')
#ax1.legend(fontsize = 20)
# adjust size of tick labels
ax1_w.tick_params(axis='both', which='major', labelsize=20)


ax1_ae.plot(eta_arr, ae_total_arr[:,0], 'b', linewidth = 3) 
# how to adjust the size and mark of the scatter points
ax1_ae.scatter(eta_arr[j], ae_total_arr[j,0], s=150 , marker = '*', c='b', label = r'$\bar{\eta}$' + ' = {:.2f}, '.format(eta_arr[j]))# + r'$\hat{A}$' + ' = {:.3f}'.format(ae_total_arr[j,0]))
#ax.set_ylim(np.min(ae_total_arr[:,0])-0.001, 0.1)

ax1_ae.set_yscale('log')
ax1_ae.grid(alpha = 0.5)
#ax1_ae.set_xlabel(r'$\bar{\eta}$', size = 28)
#ax1_ae.set_ylabel(r'$\hat{A}$', size = 28, loc = 'top')
#ax1_ae.legend(fontsize = 22)
# ax1_ae.set_title(r'$\bar{\eta}$' + ' = {:.2f}, '.format(eta_arr[j]), size = 28)

ax1_ae.set_title(r'$\hat{A}$', size = 28, color = 'b', loc = 'left', pad = 20)

ax1_ae.text(0.05, 0.15,r'$\bar{\eta}$' + ' = {:.2f}'.format(eta_arr[j]), transform=ax1_ae.transAxes, fontsize=24)

ax1_ae.tick_params(axis='both', which='major', labelsize=20)

# bounce well plot

# cross section plots 
ax1_cross.tick_params(axis='both', which='major', labelsize=20)
ax1_cross.axis('off')
# limits of axes
#   ax3.set_ylim(-0.8, 0.8)
#   ax3.set_xlim(0.8, 1.5)


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
        # elif phinorm == 0.25:
        #     label = r'$\phi={\pi}/$' + str(2 * self.nfp)
        # elif phinorm == 0.5:
        #     label = r'$\phi=\pi/$' + str(self.nfp)
        # elif phinorm == 0.75:
        #     label = r'$\phi={3\pi}/$' + str(2 * self.nfp)
        else:
            label = '_nolegend_'
        color = next(ax1._get_lines.prop_cycler)['color']
        # Plot location of the axis
        ax1_cross.plot(self.R0_func(phi), self.Z0_func(phi), marker="x", linewidth=2, label=label, color=color)
        # Plot poloidal cross-section
        ax1_cross.plot(R_2D_spline(phi), z_2D_spline(phi), color=color)
    # plt.xlabel('R (meters)')
    # plt.ylabel('Z (meters)')
    ax1_cross.legend(loc = "center", bbox_to_anchor=(0.5, 2.3), fontsize = 18)
    #plt.tight_layout()
    ax1_cross.set_aspect('equal')

plot_boundary(stel, r = 0.03)


############ 2 plot ####################

j = 17

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
    ax2.plot([-var_theta, var_theta], [B_vartheta, B_vartheta], c = colors[idx], linewidth = 0.9)

ax2.plot(vartheta, magB, c = 'black')
ax2.tick_params(axis='both', which='major', labelsize=20)
#ax2.set_ylabel('$|B|$', size = 28)
#   ax1.set_xlabel(r'$\vartheta$', size = 18)


# vmin = np.min(ae_per_lam_nor_arr[:, j, 0])  # get minimum of nth channel
# vmax = vmax=np.max(ae_per_lam_nor_arr[:, j, 0])  # get maximum of nth channel
# cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
# cbar.draw_all()

ax2_w = ax2.twinx()

ax2_w.plot(vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 3, alpha = 0.8)
ax2_w.plot(-vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 3, alpha = 0.8, label = r'$\hat{\omega}_{\alpha}$')

# find where walpha is close to zero


ax2_w.plot(vartheta_ae, 0*w_alpha_arr[:, j, 0], '-.b', linewidth = 3, alpha = 0.8)
ax2_w.plot(-vartheta_ae, 0*w_alpha_arr[:, j, 0], '-.b', linewidth = 3, alpha = 0.8, label = r'$\hat{\omega}_{\psi}$')

#ax2_w.set_ylabel(r'$\hat{\omega}_{\alpha}$', color = 'r', size = 24, loc='bottom')
ax2_w.spines['right'].set_color('r')
ax2_w.tick_params(colors='red', which='major')
# ax2.legend(fontsize = 20)
# adjust size of tick labels
ax2_w.tick_params(axis='both', which='major', labelsize=20)


ax2_ae.plot(eta_arr, ae_total_arr[:,0], 'b', linewidth = 3) 
# how to adjust the size and mark of the scatter points
ax2_ae.scatter(eta_arr[j], ae_total_arr[j,0], s=150 , marker = '*', c='b', label = r'$\bar{\eta}$' + ' = {:.2f}, '.format(eta_arr[j]))# + r'$\hat{A}$' + ' = {:.3f}'.format(ae_total_arr[j,0]))
#ax.set_ylim(np.min(ae_total_arr[:,0])-0.001, 0.1)

ax2_ae.set_yscale('log')
ax2_ae.grid(alpha = 0.5)
# ax2_ae.set_xlabel(r'$\bar{\eta}$', size = 28)
# ax2_ae.set_ylabel(r'$\hat{A}$', size = 28, loc = 'top')
# ax2_ae.legend(fontsize = 20)
# ax2_ae.set_title(r'$\hat{A}$ \quad \quad \quad' + r'$\bar{\eta}$' + ' = {:.2f}'.format(eta_arr[j]), size = 28)

ax2_ae.text(0.05, 0.15,r'$\bar{\eta}$' + ' = {:.2f}'.format(eta_arr[j]), transform=ax2_ae.transAxes, fontsize=24)

#ax2_ae.set_title(r'$\hat{A}$', size = 28, color = 'b')

ax2_ae.tick_params(axis='both', which='major', labelsize=20)

# bounce well plot

# cross section plots 
ax2_cross.tick_params(axis='both', which='major', labelsize=20)
ax2_cross.axis('off')
# limits of axes
#   ax3.set_ylim(-0.8, 0.8)
#   ax3.set_xlim(0.8, 1.5)


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
        # if phinorm == 0:
        #     label = r'$\phi$=0'
        if phinorm == 0.25:
            label = r'$\phi={\pi}/$' + str(2 * self.nfp)
        # elif phinorm == 0.5:
        #     label = r'$\phi=\pi/$' + str(self.nfp)
        # elif phinorm == 0.75:
        #     label = r'$\phi={3\pi}/$' + str(2 * self.nfp)
        else:
            label = '_nolegend_'
        color = next(ax2._get_lines.prop_cycler)['color']
        # Plot location of the axis
        ax2_cross.plot(self.R0_func(phi), self.Z0_func(phi), marker="x", linewidth=2, label=label, color=color)
        # Plot poloidal cross-section
        ax2_cross.plot(R_2D_spline(phi), z_2D_spline(phi), color=color)
    # plt.xlabel('R (meters)')
    # plt.ylabel('Z (meters)')
    ax2_cross.legend(loc = "center", bbox_to_anchor=(0.5, 1.7), fontsize = 18)
    #plt.tight_layout()
    ax2_cross.set_aspect('equal')

plot_boundary(stel, r = 0.03)


########## plot 3 ##########


j = 23

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
    ax3.plot([-var_theta, var_theta], [B_vartheta, B_vartheta], c = colors[idx], linewidth = 0.9)

ax3.plot(vartheta, magB, c = 'black')
ax3.tick_params(axis='both', which='major', labelsize=20)
#ax3.set_ylabel('$|B|$', size = 28)
#   ax1.set_xlabel(r'$\vartheta$', size = 18)


# vmin = np.min(ae_per_lam_nor_arr[:, j, 0])  # get minimum of nth channel
# vmax = vmax=np.max(ae_per_lam_nor_arr[:, j, 0])  # get maximum of nth channel
# cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
# cbar.draw_all()

ax3_w = ax3.twinx()

ax3_w.plot(vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 3, alpha = 0.8)
ax3_w.plot(-vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 3, alpha = 0.8, label = r'$\hat{\omega}_{\alpha}$')

# find where walpha is close to zero


ax3_w.plot(vartheta_ae, 0*w_alpha_arr[:, j, 0], '-.b', linewidth = 3, alpha = 0.8)
ax3_w.plot(-vartheta_ae, 0*w_alpha_arr[:, j, 0], '-.b', linewidth = 3, alpha = 0.8, label = r'$\hat{\omega}_{\psi}$')

#ax3_w.set_ylabel(r'$\hat{\omega}_{\alpha}$', color = 'r', size = 24, loc='bottom')
ax3_w.spines['right'].set_color('r')
ax3_w.tick_params(colors='red', which='major')
# ax2.legend(fontsize = 20)
# adjust size of tick labels
ax3_w.tick_params(axis='both', which='major', labelsize=20)


ax3_ae.plot(eta_arr, ae_total_arr[:,0], 'b', linewidth = 3) 
# how to adjust the size and mark of the scatter points
ax3_ae.scatter(eta_arr[j], ae_total_arr[j,0], s=150 , marker = '*', c='b', label = r'$\bar{\eta}$' + ' = {:.2f}, '.format(eta_arr[j]))# + r'$\hat{A}$' + ' = {:.3f}'.format(ae_total_arr[j,0]))
#ax.set_ylim(np.min(ae_total_arr[:,0])-0.001, 0.1)

ax3_ae.set_yscale('log')
ax3_ae.grid(alpha = 0.5)
#ax3_ae.set_xlabel(r'$\bar{\eta}$', size = 28)
# ax3_ae.set_ylabel(r'$\hat{A}$', size = 28, loc = 'top')
# ax3_ae.legend(fontsize = 22)
#ax3_ae.set_title(r'$\hat{A}$ \quad \quad \quad' + r'$\bar{\eta}$' + ' = {:.2f}'.format(eta_arr[j]), size = 28)
ax3_ae.text(0.05, 0.15,r'$\bar{\eta}$' + ' = {:.2f}'.format(eta_arr[j]), transform=ax3_ae.transAxes, fontsize=24)
ax3_ae.tick_params(axis='both', which='major', labelsize=20)

# bounce well plot

# cross section plots 
ax3_cross.tick_params(axis='both', which='major', labelsize=20)
ax3_cross.axis('off')
# limits of axes
#   ax3.set_ylim(-0.8, 0.8)
#   ax3.set_xlim(0.8, 1.5)


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
        # if phinorm == 0:
        #     label = r'$\phi$=0'
        # elif phinorm == 0.25:
        #     label = r'$\phi={\pi}/$' + str(2 * self.nfp)
        if phinorm == 0.5:
            label = r'$\phi=\pi/$' + str(self.nfp)
        # elif phinorm == 0.75:
        #     label = r'$\phi={3\pi}/$' + str(2 * self.nfp)
        else:
            label = '_nolegend_'
        color = next(ax3._get_lines.prop_cycler)['color']
        # Plot location of the axis
        ax3_cross.plot(self.R0_func(phi), self.Z0_func(phi), marker="x", linewidth=2, label=label, color=color)
        # Plot poloidal cross-section
        ax3_cross.plot(R_2D_spline(phi), z_2D_spline(phi), color=color)
    # plt.xlabel('R (meters)')
    # plt.ylabel('Z (meters)')
    ax3_cross.legend(loc = "center", bbox_to_anchor=(0.5, 1.15), fontsize = 18)
    #plt.tight_layout()
    ax3_cross.set_aspect('equal')

plot_boundary(stel, r = 0.03)


############### plot 4 ####################


j = 28

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
    ax4.plot([-var_theta, var_theta], [B_vartheta, B_vartheta], c = colors[idx], linewidth = 0.9)

ax4.plot(vartheta, magB, c = 'black')
ax4.tick_params(axis='both', which='major', labelsize=20)
#ax4.set_ylabel('$|B|$', size = 28)
ax4.set_xlabel(r'$\vartheta$', size = 28)


# vmin = np.min(ae_per_lam_nor_arr[:, j, 0])  # get minimum of nth channel
# vmax = vmax=np.max(ae_per_lam_nor_arr[:, j, 0])  # get maximum of nth channel
# cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
# cbar.draw_all()

ax4_w = ax4.twinx()

ax4_w.plot(vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 3, alpha = 0.8)
ax4_w.plot(-vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 3, alpha = 0.8, label = r'$\hat{\omega}_{\alpha}$')

# find where walpha is close to zero


ax4_w.plot(vartheta_ae, 0*w_alpha_arr[:, j, 0], '-.b', linewidth = 3, alpha = 0.8)
ax4_w.plot(-vartheta_ae, 0*w_alpha_arr[:, j, 0], '-.b', linewidth = 3, alpha = 0.8, label = r'$\hat{\omega}_{\psi}$')

#ax4_w.set_ylabel(r'$\hat{\omega}_{\alpha}$', color = 'r', size = 24, loc='bottom')
ax4_w.spines['right'].set_color('r')
ax4_w.tick_params(colors='red', which='major')
# ax2.legend(fontsize = 20)
# adjust size of tick labels
ax4_w.tick_params(axis='both', which='major', labelsize=20)


ax4_ae.plot(eta_arr, ae_total_arr[:,0], 'b', linewidth = 3) 
# how to adjust the size and mark of the scatter points
ax4_ae.scatter(eta_arr[j], ae_total_arr[j,0], s=150 , marker = '*', c='b', label = r'$\bar{\eta}$' + ' = {:.2f}, '.format(eta_arr[j]))# + r'$\hat{A}$' + ' = {:.3f}'.format(ae_total_arr[j,0]))
#ax.set_ylim(np.min(ae_total_arr[:,0])-0.001, 0.1)

ax4_ae.set_yscale('log')
ax4_ae.grid(alpha = 0.5)
ax4_ae.set_xlabel(r'$\bar{\eta}$', size = 28)
# ax4_ae.set_ylabel(r'$\hat{A}$', size = 28, loc = 'top')
# ax4_ae.legend(fontsize = 22)
# ax4_ae.set_title(r'$\hat{A}$ \quad \quad \quad' + r'$\bar{\eta}$' + ' = {:.2f}'.format(eta_arr[j]), size = 28)
# put text on pplot
ax4_ae.text(0.05, 0.15,r'$\bar{\eta}$' + ' = {:.2f}'.format(eta_arr[j]), transform=ax4_ae.transAxes, fontsize=24)
ax4_ae.tick_params(axis='both', which='major', labelsize=20)

# bounce well plot

# cross section plots 
ax4_cross.tick_params(axis='both', which='major', labelsize=20)
ax4_cross.axis('off')
# limits of axes
#   ax3.set_ylim(-0.8, 0.8)
#   ax3.set_xlim(0.8, 1.5)


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
        # if phinorm == 0:
        #     label = r'$\phi$=0'
        # elif phinorm == 0.25:
        #     label = r'$\phi={\pi}/$' + str(2 * self.nfp)
        # elif phinorm == 0.5:
        #     label = r'$\phi=\pi/$' + str(self.nfp)
        if phinorm == 0.75:
            label = r'$\phi={3\pi}/$' + str(2 * self.nfp)
        else:
            label = '_nolegend_'
        color = next(ax4._get_lines.prop_cycler)['color']
        # Plot location of the axis
        ax4_cross.plot(self.R0_func(phi), self.Z0_func(phi), marker="x", linewidth=2, label=label, color=color)
        # Plot poloidal cross-section
        ax4_cross.plot(R_2D_spline(phi), z_2D_spline(phi), color=color)
    # plt.xlabel('R (meters)')
    # plt.ylabel('Z (meters)')
    ax4_cross.legend(loc = "center", bbox_to_anchor=(0.5, 1.15), fontsize = 18)
    #plt.tight_layout()
    ax4_cross.set_aspect('equal')

plot_boundary(stel, r = 0.03)

ax4_cross.set_xlabel('cross section view \n of flux surfaces', size = 18)

# add text at bottom of plot
ax4_cross.text(0.5, -0.3, 'cross section view \n of flux surfaces', ha='center', va='center', transform=ax4_cross.transAxes, size = 18)

plt.savefig('bounce_wells.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

"""

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



# # function for doing animation

# def update_linechart(i):

#   for j in range(i+1):

#       ax.clear()
#       ax.cla()

#       ax1.clear()
#       #cax.cla()
#       ax1.cla()
#       #ax2=ax1.twinx()
#       #caxx.cla()
#       ax2.cla()
#       ax2.clear()

#       ax3.clear()
#       ax3.cla()

#       #ax1.clf()


#       # defining ae_per_lamb

#       eta = eta_arr[j]

#       stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)

#       # things for plotting the well
#       res = 5000

#       theta = np.linspace(-np.pi, np.pi, res)

#       N = stel.iota - stel.iotaN

#       phi = theta/(stel.iotaN + N)

#       vartheta = theta - N*phi

#       lamb_min = lamb_min = 1/(1 - r*eta)

#       lamb_max = lamb_max = 1/(1 + r*eta)

#       lam_arr = np.linspace(lamb_min, lamb_max, lam_res)[1:-1]

#       vartheta_ae = 2*np.arcsin(np.sqrt((((1 - lam_arr*(1 + r*eta))/(-r*eta*lam_arr)))/2))

#       magB = stel.B_mag(r=r, theta=theta, phi=phi)

#       norm = plt.Normalize()
#       color_arr = np.concatenate((ae_per_lam_arr[:, j, 0], np.array([ae_per_lam_min, ae_per_lam_max])))
#       #print(color_arr.shape)
#       colors = plt.cm.plasma(norm(color_arr))

#       # cmap = plt.get_cmap('plasma', 200)
#       # norm = mpl.colors.Normalize(vmin=np.min(ae_per_lam_nor_arr[:, j, 0]), vmax=np.max(ae_per_lam_nor_arr[:, j, 0]))
#       # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#       # sm.set_array([])

#       for idx, var_theta in enumerate(vartheta_ae):

#           B_vartheta = np.interp(var_theta, vartheta, magB)
#           ax1.plot([-var_theta, var_theta], [B_vartheta, B_vartheta], c = colors[idx], linewidth = 0.9)

#       ax1.plot(vartheta, magB, c = 'black')
#       ax1.tick_params(axis='both', which='major', labelsize=16)
#     #   ax1.set_ylabel('$|B|$', size = 18)
#     #   ax1.set_xlabel(r'$\vartheta$', size = 18)

      
#       # vmin = np.min(ae_per_lam_nor_arr[:, j, 0])  # get minimum of nth channel
#       # vmax = vmax=np.max(ae_per_lam_nor_arr[:, j, 0])  # get maximum of nth channel
#       # cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
#       # cbar.draw_all()


#       ax2.plot(vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 3, alpha = 0.8)
#       ax2.plot(-vartheta_ae, w_alpha_arr[:, j, 0], '-.r', linewidth = 3, alpha = 0.8, label = r'$\hat{\omega}_{\alpha}$')

#       # find where walpha is close to zero


#       ax2.plot(vartheta_ae, 0*w_alpha_arr[:, j, 0], '-.b', linewidth = 3, alpha = 0.8)
#       ax2.plot(-vartheta_ae, 0*w_alpha_arr[:, j, 0], '-.b', linewidth = 3, alpha = 0.8, label = r'$\hat{\omega}_{\psi}$')

#       #ax2.set_ylabel(r'$\hat{\omega}_{\alpha}$', color = 'r', size = 18)
#       ax2.spines['right'].set_color('r')
#       ax2.tick_params(colors='red', which='major')
#       # ax2.legend(fontsize = 20)
#       # adjust size of tick labels
#       ax2.tick_params(axis='both', which='major', labelsize=16)


#       ax.plot(eta_arr, ae_total_arr[:,0], 'b', linewidth = 3) 
#       # how to adjust the size and mark of the scatter points
#       ax.scatter(eta_arr[j], ae_total_arr[j,0], s=55 , c='b', label = r'$\bar{\eta}$' + ' = {:.2f}, '.format(eta_arr[j]) + r'$\hat{A}$' + ' = {:.3f}'.format(ae_total_arr[j,0]))
#       #ax.set_ylim(np.min(ae_total_arr[:,0])-0.001, 0.1)

#       ax.set_yscale('log')
#       ax.grid(alpha = 0.5)
#     #   ax.set_xlabel(r'$\bar{\eta}$', size = 18)
#     #   ax.set_ylabel(r'$\hat{A}$', size = 18)
#       ax.legend(fontsize = 17)
#       ax.tick_params(axis='both', which='major', labelsize=16)

#       # bounce well plot

#       # cross section plots 
#       ax3.tick_params(axis='both', which='major', labelsize=16)
#       ax3.axis('off')
#       # limits of axes
#     #   ax3.set_ylim(-0.8, 0.8)
#     #   ax3.set_xlim(0.8, 1.5)
#       plot_boundary(stel, r = 0.03 )

#     #   fig.tight_layout()


# num_frames = len(eta_arr)
# anim = animation.FuncAnimation(fig, update_linechart, frames = num_frames, interval = 150)
# anim.save('animate_well{}_changed.gif'.format(omn))

"""