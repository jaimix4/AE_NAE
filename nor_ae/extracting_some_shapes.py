import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qsc import Qsc
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from scipy.interpolate import interp2d, interp1d
from ae_nor_func_pyqsc import ae_nor_nae
from opt_eta_star import set_eta_star
import time
from   matplotlib        import rc
# add latex fonts to plots
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 22})
rc('text', usetex=True)

import matplotlib.colors as clr
from matplotlib.colors import LightSource
from matplotlib import cm


# def plot_boundary(self, r=0.1, ntheta=80, nphi=150, ntheta_fourier=20, nsections=8,
#          fieldlines=False, savefig=None, colormap=None, azim_default=None,
#          show=True, **kwargs):

#     x_2D_plot, y_2D_plot, z_2D_plot, R_2D_plot = self.get_boundary(r=r, ntheta=ntheta, nphi=nphi, ntheta_fourier=ntheta_fourier)
#     phi = np.linspace(0, 2 * np.pi, nphi)  # Endpoint = true and no nfp factor, because this is what is used in get_boundary()
#     R_2D_spline = interp1d(phi, R_2D_plot, axis=1)
#     z_2D_spline = interp1d(phi, z_2D_plot, axis=1)

#     ## Poloidal plot
#     phi1dplot_RZ = np.linspace(0, 2 * np.pi / self.nfp, nsections, endpoint=False)
#     # fig = plt.figure(figsize=(6, 6), dpi=80)
#     # ax  = plt.gca()
#     for i, phi in enumerate(phi1dplot_RZ):
#         phinorm = phi * self.nfp / (2 * np.pi)
#         # if phinorm == 0:
#         #     label = r'$\phi$=0'
#         # elif phinorm == 0.25:
#         #     label = r'$\phi={\pi}/$' + str(2 * self.nfp)
#         if phinorm == 0.5:
#             label = r'$\phi=\pi/$' + str(self.nfp)
#         # elif phinorm == 0.75:
#         #     label = r'$\phi={3\pi}/$' + str(2 * self.nfp)
#         else:
#             label = '_nolegend_'
#         color = next(ax3._get_lines.prop_cycler)['color']
#         # Plot location of the axis
#         ax3_cross.plot(self.R0_func(phi), self.Z0_func(phi), marker="x", linewidth=2, label=label, color=color)
#         # Plot poloidal cross-section
#         ax3_cross.plot(R_2D_spline(phi), z_2D_spline(phi), color=color)
#     # plt.xlabel('R (meters)')
#     # plt.ylabel('Z (meters)')
#     ax3_cross.legend(loc = "center", bbox_to_anchor=(0.5, 1.15), fontsize = 18)
#     #plt.tight_layout()
#     ax3_cross.set_aspect('equal')

def create_field_lines(qsc, alphas, X_2D, Y_2D, Z_2D, phimax=2*np.pi, nphi=500):
    '''
    Function to compute the (X, Y, Z) coordinates of field lines at
    several alphas, where alpha = theta-iota*varphi with (theta,varphi)
    the Boozer toroidal angles. This function relies on a 2D interpolator
    from the scipy library to smooth out the lines

    Args:
      qsc: instance of self
      alphas: array of field line labels alpha
      X_2D: 2D array for the x components of the surface
      Y_2D: 2D array for the y components of the surface
      Z_2D: 2D array for the z components of the surface
      phimax: maximum value for the field line following angle phi
      nphi: grid resolution for the output fieldline
    '''
    phi_array = np.linspace(0,phimax,nphi,endpoint=False)
    fieldline_X = np.zeros((len(alphas),nphi))
    fieldline_Y = np.zeros((len(alphas),nphi))
    fieldline_Z = np.zeros((len(alphas),nphi))
    [ntheta_RZ,nphi_RZ] = X_2D.shape
    phi1D   = np.linspace(0,2*np.pi,nphi_RZ)
    theta1D = np.linspace(0,2*np.pi,ntheta_RZ)
    X_2D_spline = interp2d(phi1D, theta1D, X_2D, kind='cubic')
    Y_2D_spline = interp2d(phi1D, theta1D, Y_2D, kind='cubic')
    Z_2D_spline = interp2d(phi1D, theta1D, Z_2D, kind='cubic')
    for i in range(len(alphas)):
        for j in range(len(phi_array)):
            phi_mod = np.mod(phi_array[j],2*np.pi)
            varphi0=qsc.nu_spline(phi_array[j])+2*phi_array[j]-phi_mod
            theta_fieldline=qsc.iota*varphi0+alphas[i]
            theta_fieldline_mod=np.mod(theta_fieldline,2*np.pi)
            fieldline_X[i,j] = X_2D_spline(phi_mod,theta_fieldline_mod)[0]
            fieldline_Y[i,j] = Y_2D_spline(phi_mod,theta_fieldline_mod)[0]
            fieldline_Z[i,j] = Z_2D_spline(phi_mod,theta_fieldline_mod)[0]
    return fieldline_X, fieldline_Y, fieldline_Z

def set_axes_equal(ax):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Args:
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def create_subplot(ax, x_2D_plot, y_2D_plot, z_2D_plot, colormap, elev=90, azim=45, dist=7, alpha=1, **kwargs):
    '''
    Construct the surface given a surface in cartesian coordinates
    x_2D_plot, y_2D_plot, z_2D_plot already with phi=[0,2*pi].
    A matplotlib figure with elements fig, ax
    must have been previously created.

    Args:
        ax: matplotlib figure instance
        x_2d_plot: 2D array for the x coordinates of the surface
        y_2d_plot: 2D array for the x coordinates of the surface
        z_2d_plot: 2D array for the x coordinates of the surface
        elev: elevation angle for the camera view
        azim: azim angle for the camera view
        distance: distance parameter for the camera view
        alpha: opacity of the surface
    '''
    ax.plot_surface(x_2D_plot, y_2D_plot, z_2D_plot, facecolors=colormap,
                    rstride=1, cstride=1, antialiased=False,
                    linewidth=0, alpha=alpha, shade=False, **kwargs)
    set_axes_equal(ax)
    ax.set_axis_off()
    ax.dist = dist
    ax.elev = elev
    ax.azim = azim

def create_subplot_mayavi(mlab, R, alphas, x_2D_plot, y_2D_plot, z_2D_plot,
                          fieldline_X, fieldline_Y, fieldline_Z,
                          Bmag, degrees_array_x, degrees_array_z, shift_array):
    '''
    Plotting routine for a mayavi figure instance that plots
    both the surface and the field lines together. The number
    of surfaces to plot is specified by the length of the
    array degrees_array_x

    Args:
      mlab: mayavi package
      R: scipy rotation vector package
      alphas: array of field line labels alpha
      x_2D_plot: 2D array for the x components of the surface
      y_2D_plot: 2D array for the y components of the surface
      z_2D_plot: 2D array for the z components of the surface
      fieldline_X: 2D array for the x components of the field line
      fieldline_Y: 2D array for the x components of the field line
      fieldline_Z: 2D array for the x components of the field line
      Bmag: 2D array for the magnetic field modulus on the (theta,phi) meshgrid
      degrees_array_x: 1D array with the rotation angles in the x direction for each surface
      degrees_array_z: 1D array with the rotation angles in the z direction for each surface
      shift_array: 1D array with a shift in the y direction for each surface
    '''
    assert len(degrees_array_x) == len(degrees_array_z) == len(shift_array)
    for i in range(len(degrees_array_x)):
        # The surfaces and field lines are rotated first in the
        # z direction and then in the x direction
        rx= R.from_euler('x', degrees_array_x[i], degrees=True)
        rz= R.from_euler('z', degrees_array_z[i], degrees=True)
        # Initialize rotated arrays
        x_2D_plot_rotated = np.zeros((x_2D_plot.shape[0],x_2D_plot.shape[1]))
        y_2D_plot_rotated = np.zeros((x_2D_plot.shape[0],x_2D_plot.shape[1]))
        z_2D_plot_rotated = np.zeros((x_2D_plot.shape[0],x_2D_plot.shape[1]))
        fieldline_X_rotated = np.zeros((fieldline_X.shape[0],fieldline_X.shape[1]))
        fieldline_Y_rotated = np.zeros((fieldline_X.shape[0],fieldline_X.shape[1]))
        fieldline_Z_rotated = np.zeros((fieldline_X.shape[0],fieldline_X.shape[1]))
        # Rotate surfaces
        for th in range(x_2D_plot.shape[0]):
            for ph in range(x_2D_plot.shape[1]):
                [x_2D_plot_rotated[th,ph], y_2D_plot_rotated[th,ph], z_2D_plot_rotated[th,ph]] = rx.apply(rz.apply(np.array([x_2D_plot[th,ph], y_2D_plot[th,ph], z_2D_plot[th,ph]])))
        # Rotate field lines
        for th in range(fieldline_X.shape[0]):
            for ph in range(fieldline_X.shape[1]):
                [fieldline_X_rotated[th,ph], fieldline_Y_rotated[th,ph], fieldline_Z_rotated[th,ph]] = rx.apply(rz.apply(np.array([fieldline_X[th,ph], fieldline_Y[th,ph], fieldline_Z[th,ph]])))
        # Plot surfaces
        mlab.mesh(x_2D_plot_rotated, y_2D_plot_rotated-shift_array[i], z_2D_plot_rotated, scalars=Bmag, colormap='viridis')
        # Plot field lines
        for j in range(len(alphas)):
            mlab.plot3d(fieldline_X_rotated[j], fieldline_Y_rotated[j]-shift_array[i], fieldline_Z_rotated[j], color=(0,0,0), line_width=0.001, tube_radius=0.005)


def plot_boundary_jaime(self, r=0.1, ntheta=80, nphi=150, ntheta_fourier=20, nsections=8,
         fieldlines=False, savefig=None, colormap=None, azim_default=None,
         show=True, **kwargs):
    """
    Plot the boundary of the near-axis configuration. There are two main ways of
    running this function.

    If ``fieldlines=False`` (default), 2 matplotlib figures are generated:

        - A 2D plot with several poloidal planes at the specified radius r with the
          corresponding location of the magnetic axis.

        - A 3D plot with the flux surface and the magnetic field strength
          on the surface.

    If ``fieldlines=True``, both matplotlib and mayavi are required, and
    the following 2 figures are generated:

        - A 2D matplotlib plot with several poloidal planes at the specified radius r with the
          corresponding location of the magnetic axis.

        - A 3D mayavi figure with the flux surface the magnetic field strength
          on the surface and several magnetic field lines.

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the poloidal angle.
      nphi   (int): Number of grid points to plot in the toroidal angle.
      ntheta_fourier (int): Resolution in the Fourier transform to cylindrical coordinates
      nsections (int): Number of poloidal planes to show.
      fieldlines (bool): Specify if fieldlines are shown. Using mayavi instead of matplotlib due to known bug https://matplotlib.org/2.2.2/mpl_toolkits/mplot3d/faq.html
      savefig (str): Filename prefix for the png files to save.
        Note that a suffix including ``.png`` will be appended.
        If ``None``, no figure files will be saved.
      colormap (cmap): Custom colormap for the 3D plots
      azim_default: Default azimuthal angle for the three subplots in the 3D surface plot
      show: Whether or not to call the matplotlib/mayavi ``show()`` command.
      kwargs: Any additional key-value pairs to pass to matplotlib's plot_surface.

    This function generates plots similar to the ones below:

    .. image:: 3dplot1.png
       :width: 200

    .. image:: 3dplot2.png
       :width: 200

    .. image:: poloidalplot.png
       :width: 200
    """

    fig = plt.figure(constrained_layout=False, figsize=(24/1.5, 7/1.5))
    gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1, 0.04, 0.7], height_ratios = [0.2, 1, 0.2], hspace = -0.1)
    ax = fig.add_subplot(gs[1, 2])

    x_2D_plot, y_2D_plot, z_2D_plot, R_2D_plot = self.get_boundary(r=r, ntheta=ntheta, nphi=nphi, ntheta_fourier=ntheta_fourier)
    phi = np.linspace(0, 2 * np.pi, nphi)  # Endpoint = true and no nfp factor, because this is what is used in get_boundary()
    R_2D_spline = interp1d(phi, R_2D_plot, axis=1)
    z_2D_spline = interp1d(phi, z_2D_plot, axis=1)
    
    ## Poloidal plot
    phi1dplot_RZ = np.linspace(0, 2 * np.pi / self.nfp, nsections, endpoint=False)
    # fig = plt.figure(figsize=(6, 6), dpi=80)
    #ax  = plt.gca()
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
        ax.plot(self.R0_func(phi), self.Z0_func(phi), marker="x", linewidth=2, label=label, color=color)
        # Plot poloidal cross-section
        ax.plot(R_2D_spline(phi), z_2D_spline(phi), color=color)
    ax.set_xlabel('R [1]')
    ax.set_ylabel('Z [1]')
    # put y ticks and label in the right side
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.legend(ncols = 2, loc='upper center', bbox_to_anchor=(0.5, 1.32), fancybox=True, fontsize = 14)
    #ax.legend(ncols = 1, loc='center', bbox_to_anchor=(-0.5, 0.5), fancybox=True, fontsize = 16)
    #plt.tight_layout()
    ax.set_aspect('equal')
    if savefig != None:
        fig.savefig(savefig + '_poloidal.pdf')

    ## 3D plot
    # Set the default azimuthal angle of view in the 3D plot
    # QH stellarators look rotated in the phi direction when
    # azim_default = 0
    if azim_default == None:
        if self.helicity == 0:
            azim_default = 0
        else:
            azim_default = 45

    
    # Define the magnetic field modulus and create its theta,phi array
    # The norm instance will be used as the colormap for the surface
    theta1D = np.linspace(0, 2*np.pi, ntheta)
    phi1D = np.linspace(0, 2 * np.pi, nphi)
    phi2D, theta2D = np.meshgrid(phi1D, theta1D)
    # Create a color map similar to viridis 
    Bmag = self.B_mag(r, theta2D, phi2D)
    norm = clr.Normalize(vmin=Bmag.min(), vmax=Bmag.max())
    if fieldlines==False:
        if colormap==None:
            # Cmap similar to quasisymmetry papers
            # cmap = clr.LinearSegmentedColormap.from_list('qs_papers',['#4423bb','#4940f4','#2e6dff','#0097f2','#00bacc','#00cb93','#00cb93','#7ccd30','#fbdc00','#f9fc00'], N=256)
            cmap = cm.viridis
            # Add a light source so the surface looks brighter
            ls = LightSource(azdeg=0, altdeg=10)
            cmap_plot = ls.shade(Bmag, cmap, norm=norm)
        # Create the 3D figure and choose the following parameters:
        # gsParams: extension in the top, bottom, left right directions for each subplot
        # elevParams: elevation (distance to the plot) for each subplot

        #fig = plt.figure(constrained_layout=False, figsize=(4.5, 4.5))


        # gsParams = [[1.02,-0.3,0.,0.85], [1.09,-0.3,0.,0.85], [1.12,-0.15,0.,0.85]]
        # elevParams = [90, 30, 5]

        # for i in range(len(gsParams)):
        #     gs = fig.add_gridspec(nrows=3, ncols=1,
        #                           top=gsParams[i][0], bottom=gsParams[i][1],
        #                           left=gsParams[i][2], right=gsParams[i][3],
        #                           hspace=0.0, wspace=0.0)
        #     ax = fig.add_subplot(gs[i, 0], projection='3d')
        #     create_subplot(ax, x_2D_plot, y_2D_plot, z_2D_plot, cmap_plot, elev=elevParams[i], azim=azim_default, **kwargs)

        #gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 0.05])
                                #   top=gsParams[i][0], bottom=gsParams[i][1],
                                #   left=gsParams[i][2], right=gsParams[i][3],
                                #   hspace=0.0, wspace=0.0)
        ax = fig.add_subplot(gs[:, 0], projection='3d')



        create_subplot(ax, x_2D_plot, y_2D_plot, z_2D_plot, cmap_plot, elev=40, azim=azim_default, **kwargs)

        xlm=ax.get_xlim3d() #These are two tupples
        ylm=ax.get_ylim3d() #we use them in the next
        zlm=ax.get_zlim3d() #graph to reproduce the magnification from mousing

        ax.view_init(elev=55, azim=0) #Reproduce view
        ax.set_xlim3d(-0.72,0.72)     #Reproduce magnification
        ax.set_ylim3d(-0.72,0.72)     #...
        ax.set_zlim3d(-0.72,0.72)     #...

        cbar_ax  = fig.add_subplot(gs[1, 1])  # Colorbar subplot for plots 1 and 2

        m = cm.ScalarMappable(cmap=cmap, norm=norm)
        m.set_array([])
        cbar = plt.colorbar(m, cax=cbar_ax)
        cbar.ax.set_title(r'$|B| [T]$', fontsize=20, pad=25)



        ax.text2D(0.05, 0.95, r'$\bar{\eta}_{*}$' + ' = {:.2f}'.format(eta_star), transform=ax.transAxes, fontsize=22)


        # Save figure
        if savefig != None:
            fig.savefig(savefig + '3D.png')
        if show:
            # Show figures
            plt.savefig('simple_qa_THESIS.png', dpi=300, bbox_inches='tight')
            plt.show()
            # Close figures
            plt.close()


    else:
        ## X, Y, Z arrays for the field lines
        # Plot different field lines corresponding to different alphas
        # where alpha=theta-iota*varphi with (theta,varphi) the Boozer angles
        #alphas = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        alphas = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        # Create the field line arrays
        fieldline_X, fieldline_Y, fieldline_Z = create_field_lines(self, alphas, x_2D_plot, y_2D_plot, z_2D_plot)
        # Define the rotation arrays for the subplots
        degrees_array_x = [0., -66., 81.] # degrees for rotation in x
        degrees_array_z = [azim_default, azim_default, azim_default] # degrees for rotation in z
        shift_array   = [-1.0, 0.7, 1.8]
        # Import mayavi and rotation packages (takes a few seconds)
        from mayavi import mlab
        from scipy.spatial.transform import Rotation as R
        if show:
            # Show RZ plot
            plt.show()
            # Close RZ plot
            plt.close()
        # Create 3D figure
        fig = mlab.figure(bgcolor=(1,1,1), size=(430,720))
        # Create subplots
        create_subplot_mayavi(mlab, R, alphas, x_2D_plot, y_2D_plot, z_2D_plot,
                              fieldline_X, fieldline_Y, fieldline_Z,
                              Bmag, degrees_array_x, degrees_array_z, shift_array)
        # Create a good camera angle
        mlab.view(azimuth=0, elevation=0, distance=8.5, focalpoint=(-0.15,0,0), figure=fig)
        # Create the colorbar and change its properties
        cb = mlab.colorbar(orientation='vertical', title='|B| [T]', nb_labels=7)
        cb.scalar_bar.unconstrained_font_size = True
        cb.label_text_property.font_family = 'times'
        cb.label_text_property.bold = 0
        cb.label_text_property.font_size=24
        cb.label_text_property.color=(0,0,0)
        cb.title_text_property.font_family = 'times'
        cb.title_text_property.font_size=34
        cb.title_text_property.color=(0,0,0)
        cb.title_text_property.bold = 1
        # Save figure
        if savefig != None:
            mlab.savefig(filename=savefig+'3D_fieldlines.png', figure=fig)
        if show:
            # Show mayavi plot
            mlab.show()
            # Close mayavi plots
            mlab.close(all=True)



# A configuration is defined with B0, etabar, field periods, Fourier components and eta bar

B0 = 1        # [T] strength of the magnetic field on axis

nfp = 3       # field periods

eta = -5   # eta bar

##########################################################################################

# these quantities are provided in [m]

rc = [1, 0.045]
zs = [0, 0.045]


stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=-0.1, B0=B0)
eta_star = set_eta_star(stel)
stel.plot_boundary(fieldlines = True)

# plot_boundary_jaime(stel, fieldlines=False)

# stel.flux_tube()


"""

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
N = 6
#omn_arr = np.linspace(-20, -1, N)
# omn_arr = np.geomspace(1, 10, N)
omn_arr = [0, 1, 2, 2, 10, 5]

# omt_arr = np.geomspace(1, 10, N)
omt_arr = [1, 0, 3, 2, 10, 5]

idx_eta_dagger_arr = np.zeros(N).astype(int)

# arrays for plotting
N_eta = 100
eta_arr = np.geomspace(-700, -0.01, N_eta)

# arrays of r
N_r = 50
r_arr = np.geomspace(1e-6, 1e-3, N_r)

# empty array with shape N by N_eta
ae_total_arr = np.empty((N, N_r))
ae_nor_total_arr = np.empty((N, N_r))

eta_arr_4plot = -1*eta_arr


# fig, ax = plt.subplots(figsize = (12/1.5,7.5/1.5))

fig = plt.figure(figsize = (12*2/1.5,7.5*2/1.5))

plt.tight_layout()

grid = plt.GridSpec(2, 2, wspace = 0.2, hspace = 0.45, width_ratios=[1, 1], height_ratios=[1, 1])


eta_arr = [-2, -1.5, -0.82, -0.5][::-1]

ax_loc = [plt.subplot(grid[0, 0]), plt.subplot(grid[0, 1]), plt.subplot(grid[1, 0]), plt.subplot(grid[1, 1])]


for idx_eta, eta in enumerate(eta_arr):

    for idx_omn, omn in enumerate(omn_arr):
        omt = omt_arr[idx_omn]
        for idx, r in enumerate(r_arr):

            stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)
            ae_total_arr[idx_omn, idx], ae_nor_total_arr[idx_omn, idx] = ae_nor_nae(stel, r, lam_res, Delta_r, a_r, omn, omt, plot = False)[3:]
    
        #idx_eta_dagger_arr[idx_omt] = np.argmax(ae_nor_total_arr[idx_omt, :])

        ax_loc[idx_eta].plot(r_arr, ae_nor_total_arr[idx_omn, :], linewidth = 3, label=r'$\hat{\omega}_{n}$' + ' = {} '.format(omn) + ', ' + r'$\hat{\omega}_{T}$' + ' = {} '.format(omt))# + r'$\bar{\eta}_{\dag}$' + ' = {:.3f}'.format(eta_arr[idx_eta_dagger_arr[idx_omt]]))
        
        # calculate slope of log(r_arr) and log(ae_nor_total_arr)
        slope, intercept = np.polyfit(np.log10(r_arr), np.log10(ae_nor_total_arr[idx_omn, :]), 1)
        print(slope)

    # figure power law
    #plt.plot(r_arr, 1e-3*r_arr**(-1.5), '--', linewidth = 4, label=r'$\hat{\omega}_{n}$' + ' = {:.2f} '.format(omn))# + r'$\bar{\eta}_{\dag}$' + ' = {:.3f}'.format(eta_arr[idx_eta_dagger_arr[idx_omt]]))
    
    #plt.scatter(eta_arr_4plot[idx_eta_dagger_arr[idx_omt]], ae_nor_total_arr[idx_omt, idx_eta_dagger_arr[idx_omt]])

#plt.title('$B_0$ and axis shape is irrelevant \n' + r'$\hat{\omega}_{T}$' + ' = {} , '.format(omt) + r'$\bar{\eta}$' + ' = {}'.format(eta), size = 18)

# plt.axis([0, 10, 0, 10])
# put text in figure
    ax_loc[idx_eta].text(0.05, 0.85, r'$\bar{\eta}$' + ' = {}'.format(eta), transform=ax_loc[idx_eta].transAxes, fontsize=22)

    ax_loc[idx_eta].text(0.60, 0.05, r'$\hat{A} \: \propto r^{0.5}$', transform=ax_loc[idx_eta].transAxes, fontsize=22)
    # ax3.text(0.05, 0.10, r'$N_{\rm fp} = $' + ' {}'.format(nfp), transform=ax3.transAxes, fontsize=20)

    # figure power law

    # put a text on figure
    ax_loc[idx_eta].set_yscale('log')
    ax_loc[idx_eta].set_xscale('log')

    if idx_eta == 0:

        #ax_loc[idx_eta].set_xlabel(r'$r$', size = 28)
        ax_loc[idx_eta].set_ylabel(r'$\hat{A}$', size = 28)

    elif idx_eta == 2:

        ax_loc[idx_eta].set_xlabel(r'$r$', size = 28)
        ax_loc[idx_eta].set_ylabel(r'$\hat{A}$', size = 28)

    elif idx_eta == 3:

        ax_loc[idx_eta].set_xlabel(r'$r$', size = 28)
        #ax_loc[idx_eta].set_ylabel(r'$\hat{A}$', size = 28)


    # put legend outside figure
    


    ax_loc[idx_eta].grid(alpha = 0.5)

ax_loc[idx_eta].legend(loc='center left', bbox_to_anchor=(-1, 1.18), fontsize = 20, ncols = 3)

# title for grid whole figure



plt.savefig('THESIS_r_ae_nor_eta_{}_now.png'.format(-eta), bbox_inches='tight', dpi = 300)
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


"""