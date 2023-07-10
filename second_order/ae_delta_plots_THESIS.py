import numpy as np
from qsc import Qsc
from AEpy import ae_routines as ae
from   matplotlib        import rc
import   matplotlib.pyplot  as plt
import matplotlib as mpl
import sys
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_nor_func_pyqsc import ae_nor_nae
from opt_eta_star import set_eta_star
from qs_shape_opt import choose_Z_axis
import time
import multiprocessing as mp
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from   matplotlib        import rc
# add latex fonts to plots
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 18})
rc('text', usetex=True)



# def opt_stel_func(a,b):
#     stel = Qsc(rc=[1, a, b], zs=[0, a, b], B0 = 1.0, nfp=nfp, order='r1', nphi = 61)
#     num_iters = choose_Z_axis(stel, max_num_iter=50)
#     return stel

from scipy.interpolate import interp2d, interp1d
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

        eta_star = self.etabar

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


def ae_computations(idx):

    # if rcrit_arr[idx] < 1e-4 or rcrit_arr[idx] > 1e5:

    #     return (np.nan, np.nan)

    # idx = (idx_b, idx_a)

    stel = Qsc(rc=[1, a_mesh[idx], b_mesh[idx]], zs=[0.0, a_z_arr[idx], b_z_arr[idx]], \
        nfp=nfp, etabar=eta_arr[idx], B0=B0, order = "r2", B2c = B2c_arr[idx], nphi=nphi)
    stel.spsi = 1
    stel.calculate()
    alpha = 1.0
    stel.r = r 
    try:
        NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=alpha, N_turns=3, nphi=nphi,
                                lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
        NAE_AE.calc_AE(omn=stel.spsi*omn,omt=stel.spsi*omt,omnigenous=omnigenous)
        NAE_AE.plot_AE_per_lam(save=True)
        ae_second = NAE_AE.ae_tot
    except:
        print('could not compute AE')
        ae_second = np.nan

    ae_first = ae_nor_nae(stel, r, lam_res, 1, 1, omn, omt, plot = False)[-1]

    return (ae_first, ae_second)

####################
# ae computations parameters

nphi = int(1e3+1)
lam_res = 2001
omn = 3
omt = 0
omnigenous = False

# r = 0.0001
# r = 0.001
r = 0.01
r = 0.02

# r = 0.0001

# ok I have to do r = 0.02

####################

N = 150
# nfp = 3
B0 = 1

#########################################################

# nfp = 3

# # NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
# a_arr = np.linspace(0.0001, 0.3, N)
# b_arr = np.linspace(-0.06, 0.06, N)

# nfp = 2

# # NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 2
# a_arr = np.linspace(0.0001, 0.6, N)
# b_arr = np.linspace(-0.13, 0.13, N)

nfp = 4

# NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 4
a_arr = np.linspace(0.0001, 0.20, N)
b_arr = np.linspace(-0.0347, 0.0347, N)

#########################################################

# NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
# a_arr = np.linspace(0.0001, 0.3, N)
# b_arr = np.linspace(-0.06, 0.06, N)

# mesh grid a_arr and b_arr
# a_mesh, b_mesh = np.meshgrid(a_arr, b_arr)


a_mesh = np.load('shapes_2nd/a_arr_N_{}_nfp_{}.npy'.format(N, nfp))
b_mesh = np.load('shapes_2nd/b_arr_N_{}_nfp_{}.npy'.format(N, nfp))
a_z_arr = np.load('shapes_2nd/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
b_z_arr = np.load('shapes_2nd/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
eta_arr = np.load('shapes_2nd/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp))
B2c_arr = np.load('shapes_2nd/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp))
delB20_arr = np.load('shapes_2nd/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp))
rcrit_arr = np.load('shapes_2nd/rcrit_arr_N_{}_nfp_{}.npy'.format(N, nfp))

# arrays to fill 

ae_first_arr = np.zeros_like(a_mesh)
ae_second_arr = np.zeros_like(a_mesh)

# shape_mesh = a_mesh.shape
# a_z_arr = np.full(shape_mesh, np.nan)
# b_z_arr = np.full(shape_mesh, np.nan)
# eta_arr = np.full(shape_mesh, np.nan)
# B2c_arr = np.full(shape_mesh, np.nan)
# delB20_arr = np.full(shape_mesh, np.nan)
# rcrit_arr   = np.empty_like(a_mesh)

##########################################################################################
# finding a curve to try some things

a_fit_arr = [0.0001, 0.0227, 0.0505, 0.0946, 0.1817, 0.2012, 0.2740, 0.2983]
b_fit_arr = [0.01163, 2e-5, 0.00091, 0.00592, 0.01058, 0.01372, 0.00662, -0.04910]

curve = np.polyfit(a_fit_arr, b_fit_arr, 7)

# use the curve to write a lambda function that plots this curve
curve = np.poly1d(curve)

# get the b_arr closest to the curve
b_fit_curve_arr = curve(a_arr)

# get the all the index of b_mesh that are the closes to b_fit_curve_arr
idx_curve = []
for i in range(len(b_fit_curve_arr)):
    idx_curve.append((i, np.argmin(np.abs(b_arr - b_fit_curve_arr[i]))))
# idx = np.array(idx_cruve)


# ae_first_curve_arr = np.zeros_like(a_arr)
# ae_second_curve_arr = np.zeros_like(a_arr)

# print(idx_curve)

##########################################################################################


####################

if __name__ == "__main__":

    try:

        ae_first_arr = np.load('ae_shapes_2nd/ae_first_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))
        ae_second_arr = np.load('ae_shapes_2nd/ae_second_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))
        print('data loaded')

    except:
        print('data not found, computing it')
        num_cores = 9 #mp.cpu_count()
        pool = mp.Pool(num_cores)
        print('Number of cores used: {}'.format(num_cores))


        # now loop over a b
        print('computing ae for optimise stellarators')
        start_time = time.time()
        output_list = pool.starmap(ae_computations, [(idx, ) for idx, _ in np.ndenumerate(a_mesh)])
        # output_list = pool.starmap(ae_computations, [(idx) for idx, _ in np.ndenumerate(a_mesh)])
        # output_list = pool.starmap(ae_computations, [idx for idx in idx_curve])
        print("data generated in       --- %s seconds ---" % (time.time() - start_time))
        # close pool
        pool.close()
        # transfer data to arrays
        list_idx = 0

        for idx, _ in np.ndenumerate(a_mesh):
        # for idx in idx_curve: 

            ae_tuple = output_list[list_idx]

            ae_first_arr[idx] = ae_tuple[0]
            ae_second_arr[idx] = ae_tuple[1]

            # ae_first_curve_arr[list_idx] = ae_tuple[0]
            # ae_second_curve_arr[list_idx] = ae_tuple[1]


            list_idx = list_idx + 1

        # np.save('shapes_2nd/a_arr_N_{}_nfp_{}.npy'.format(N, nfp), a_mesh)
        # np.save('shapes_2nd/b_arr_N_{}_nfp_{}.npy'.format(N, nfp), b_mesh)
        # np.save('shapes_2nd/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp), a_z_arr)
        # np.save('shapes_2nd/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp), b_z_arr)
        # np.save('shapes_2nd/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp), eta_arr)
        # np.save('shapes_2nd/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp), B2c_arr)
        # np.save('shapes_2nd/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp), delB20_arr)
        # np.save('shapes_2nd/rcrit_arr_N_{}_nfp_{}.npy'.format(N, nfp), rcrit_arr)

        # np.save('shapes_2nd/ae_first_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}.npy'.format(N, nfp, r, omn, omt), ae_first_arr)
        # np.save('shapes_2nd/ae_second_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}.npy'.format(N, nfp, r, omn, omt), ae_second_arr)

        # print(ae_first_curve_arr)
        # print(ae_second_curve_arr)

        np.save('ae_shapes_2nd/ae_first_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt), ae_first_arr)
        np.save('ae_shapes_2nd/ae_second_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt), ae_second_arr)

    # fig = plt.figure(figsize = (7, 6))

    # plt.plot(a_arr, ae_first_curve_arr, 'r*-', label = 'AE first')
    # plt.plot(a_arr, ae_second_curve_arr, 'b-', label = 'AE second')
    # plt.title('AE at r = {}, omn = {}, omt = {}'.format(r, omn, omt))
    # plt.legend()

    fig = plt.figure(figsize=(21/1.5, 7/1.5))
    #grid = gridspec.GridSpec(1, 3, wspace=0.3)

    grid = gridspec.GridSpec(2, 3, height_ratios=[1, 0.05], wspace=0.3, hspace=0.55, width_ratios=[1, 1, 1])

    

    # make negatives values of ae_second_arr to be nan
    ae_second_arr[ae_second_arr < 0] = np.nan

    # getting rid of values which r_crit is smaller than r
    # since they are "not valid for that ordering"
    ae_first_arr[rcrit_arr < r] = np.nan
    ae_second_arr[rcrit_arr < r] = np.nan

    # make nan values of ae_second_arr to also be nan on ae_first_arr

    ae_first_arr[np.isnan(ae_second_arr)] = np.nan

    #ae_first_arr[ae_second_arr == np.nan] = np.nan



    # print(np.min(ae_second_arr))
    # ae_second_arr[ae_second_arr > 2] = np.nan
    # # min value of ae_second_arr excluding nan
    # ae_first_arr[ae_first_arr < 1e-5] = np.nan
    # print(ae_second_arr)
    # minimum of array excluding nan

    # array with non nan values of ae_first
    
    # print(len(ae_first_arr[~np.isnan(ae_first_arr)]))


    # for stel in valid_arr:

    ############# plots ae first and second, delta #############


    ax1 = plt.subplot(grid[0, 0])

    ax2 = plt.subplot(grid[0, 1])

    ax3 = plt.subplot(grid[0, 2])

    #levels = np.linspace(np.nanmin(ae_first_arr), np.nanmax(ae_first_arr), 200)

    # using max and minimum of ae_second_arr to make the levels
    
    if r == 0.0001:
        levels = np.linspace(0.0001, 0.0054, 200)
    elif r == 0.01:
        levels = np.linspace(0.006, 0.06, 200)
    else:
        levels = 200
    # plot at first order

    ax1.contourf(a_mesh, b_mesh, ae_first_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
    # plt.colorbar()
    # ax1.set_title('AE first at r = {}, omn = {}, omt = {}'.format(r, omn, omt))
    ax1.set_title(r'$\hat{A}_{L}$', fontsize = 28)
    ax1.set_xlabel(r'$a$', fontsize = 28)
    #ax1.set_ylabel(r'$b$', fontsize = 28)

    # add text to plot
    ax1.text(-0.15, 1.12, r'$N_{\rm fp}$' + ' = {}'.format(nfp), fontsize = 18, transform=ax1.transAxes, color = 'gray')
    ax1.text(-0.14, 1.02, r'$r$' + ' = {}'.format(r), fontsize = 18, transform=ax1.transAxes, color = 'gray')


    ax = ax1

    # set number of ticks to 2 decimal places
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    ######### adding lines #########

    if nfp == 2:

        a_arr = np.linspace(0.0001, 0.6, N)
        line_1 = lambda a: (-0.0612/0.22)*a - 0.0612
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.1171 + 0.0612)/0.6)*a - 0.0612
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.107, 0.269, 0.446, 0.512, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0584, 0.0588, 0.0604, 0.0695, 0.0898, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.45, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.6)
        ax.set_ylim(-0.13, 0.13)


    elif nfp == 3:

        a_arr = np.linspace(0.0001, 0.3, N)
        line_1 = lambda a: (-0.0280/0.108)*a - 0.0280
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0538 + 0.0280)/0.3)*a - 0.0280
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0591, 0.1433, 0.2211, 0.2677, 0.3] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0267, 0.0267, 0.0285, 0.0319, 0.0437, 0.0526] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.2285, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.3)
        ax.set_ylim(-0.06, 0.06)


    elif nfp == 4:

        a_arr = np.linspace(0.0001, 0.20, N)
        line_1 = lambda a: (-0.0160/0.064)*a - 0.0160
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0338 + 0.0160)/0.1891)*a - 0.0160
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0381, 0.0724, 0.1345, 0.1659, 0.1927] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0146, 0.0155, 0.0157, 0.0191, 0.0273, 0.0342] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.1368, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)


        ax.set_xlim(0.0001, 0.20)
        ax.set_ylim(-0.0347, 0.0347)



    #plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

    #fig = plt.figure()
    # levels = np.linspace(np.nanmin(ae_second_arr), np.nanmax(ae_second_arr), 200)

    ax2_plot = ax2.contourf(a_mesh, b_mesh, ae_second_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
    ax2.set_title(r'$\hat{A}$', fontsize = 28)
    #ax2.set_xlabel(r'$s$', fontsize = 28)
    ax2.set_xlabel(r'$a$', fontsize = 28)

    ax = ax2

    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    if nfp == 2:

        a_arr = np.linspace(0.0001, 0.6, N)
        line_1 = lambda a: (-0.0612/0.22)*a - 0.0612
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.1171 + 0.0612)/0.6)*a - 0.0612
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.107, 0.269, 0.446, 0.512, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0584, 0.0588, 0.0604, 0.0695, 0.0898, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.45, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.6)
        ax.set_ylim(-0.13, 0.13)


    elif nfp == 3:

        a_arr = np.linspace(0.0001, 0.3, N)
        line_1 = lambda a: (-0.0280/0.108)*a - 0.0280
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0538 + 0.0280)/0.3)*a - 0.0280
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0591, 0.1433, 0.2211, 0.2677, 0.3] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0267, 0.0267, 0.0285, 0.0319, 0.0437, 0.0526] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.2285, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.3)
        ax.set_ylim(-0.06, 0.06)


    elif nfp == 4:

        a_arr = np.linspace(0.0001, 0.20, N)
        line_1 = lambda a: (-0.0160/0.064)*a - 0.0160
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0338 + 0.0160)/0.1891)*a - 0.0160
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0381, 0.0724, 0.1345, 0.1659, 0.1927] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0146, 0.0155, 0.0157, 0.0191, 0.0273, 0.0342] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.1368, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)


        ax.set_xlim(0.0001, 0.20)
        ax.set_ylim(-0.0347, 0.0347)


    #ax2.set_xlabel(r'$\hat{\omega}_{\alpha}, \: \hat{\omega}_{\psi}$', fontsize = 28)

    cax = fig.add_subplot(grid[1, :2])  # Colorbar subplot for plots 1 and 2
    cbar = fig.colorbar(ax2_plot, cax=cax, orientation='horizontal')
    cbar.ax.locator_params(nbins=8)

    # Set the number of ticks for the color bar
    # num_ticks = 6  # Set the desired number of ticks
    # cbar.locator = ticker.MaxNLocator(num_ticks)  # Set the locator
    # cbar.update_ticks()  # Update the ticks on the color bar
    
    # add scientific notation to colorbar
    cbar.ax.ticklabel_format(style='sci', scilimits=(-2,-2), axis='both')
    # move decimal of scientific notation to the left

    cbar.ax.yaxis.set_offset_position('left')                         
    cbar.update_ticks()

    # set number of decimals points in ticks of color bar to 2

    cbar.ax.locator_params(nbins=7)

    # put scaling of scientific notation inside colorbar

    

    #cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) 

    # plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)
    # plt.savefig('first_ae_{}.png'.format(r), dpi = 300)

    delta_ae = (ae_second_arr/ae_first_arr - 1)#/r
    delta_ae[rcrit_arr < r] = np.nan
    # print(len(ae_first_arr[delta_ae < 0]))

    # index of minimum values of delta_ae in order

    # min_delta_ae = np.argnanmin(delta_ae)
    
    # provide an ordered array of the indices of the minimum values of delta_ae
    # in order to plot the minimum values of delta_ae in order
    # print(np.argsort(delta_ae))
    # print(np.argsort(delta_ae)[min_delta_ae])
    
    # print(np.argmin(delta_ae))

    # make a flat copy of delta_ae
    # delta_ae_flat = delta_ae.flatten()

    # arg_min_delta_ae = np.argsort(delta_ae_flat)
    # print(arg_min_delta_ae)

    # print(np.argmin(delta_ae))

    # filters of delta_ae

    # delta_ae[delta_ae > 0] = np.nan
    # delta_ae[delta_ae < -0.3] = np.nan
    # delta_ae[a_mesh < 0.15] = np.nan


       # index of minimum value on delta_ae
    arg_min_delta_ae = np.nanargmin(delta_ae)

    arg_min_delta_ae = np.unravel_index(np.nanargmin(delta_ae), delta_ae.shape)

    print(arg_min_delta_ae)

    print(delta_ae[arg_min_delta_ae])


    # fig = plt.figure()
    # levels = np.linspace(-20.0, 20.0, 300)
    # levels = np.linspace(-0.5, 0.5, 300)

    #levels = 300
    levels = np.linspace(-0.6, 0.6, 300)
    ax3_plot = ax3.contourf(a_mesh, b_mesh, delta_ae, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
    ax3.set_title(r'$\delta\hat{A}$', fontsize = 28)
    #plt.title('(AE2nd / AE1st - 1) ,  at r = {}, omn = {}, omt = {}'.format(r, omn, omt))
    #plt.colorbar()
    ax3.set_xlabel(r'$a$', fontsize = 28)
    ax3.set_ylabel(r'$b$', fontsize = 28)

    # put y label on the right axis
    ax3.yaxis.set_label_position("right")

    ax = ax3

    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    if nfp == 2:

        a_arr = np.linspace(0.0001, 0.6, N)
        line_1 = lambda a: (-0.0612/0.22)*a - 0.0612
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.1171 + 0.0612)/0.6)*a - 0.0612
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.107, 0.269, 0.446, 0.512, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0584, 0.0588, 0.0604, 0.0695, 0.0898, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.45, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.6)
        ax.set_ylim(-0.13, 0.13)


    elif nfp == 3:

        a_arr = np.linspace(0.0001, 0.3, N)
        line_1 = lambda a: (-0.0280/0.108)*a - 0.0280
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0538 + 0.0280)/0.3)*a - 0.0280
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0591, 0.1433, 0.2211, 0.2677, 0.3] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0267, 0.0267, 0.0285, 0.0319, 0.0437, 0.0526] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.2285, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.3)
        ax.set_ylim(-0.06, 0.06)


    elif nfp == 4:

        a_arr = np.linspace(0.0001, 0.20, N)
        line_1 = lambda a: (-0.0160/0.064)*a - 0.0160
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0338 + 0.0160)/0.1891)*a - 0.0160
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0381, 0.0724, 0.1345, 0.1659, 0.1927] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0146, 0.0155, 0.0157, 0.0191, 0.0273, 0.0342] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.1368, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)


        ax.set_xlim(0.0001, 0.20)
        ax.set_ylim(-0.0347, 0.0347)

    cax = fig.add_subplot(grid[1, 2])  # Colorbar subplot for plots 1 and 2
    cbar = fig.colorbar(ax3_plot, cax=cax, orientation='horizontal')
    cbar.ax.locator_params(nbins=5)
    #plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

    # Set the number of ticks for the color bar
    # num_ticks = 5  # Set the desired number of ticks
    # cbar.locator = ticker.MaxNLocator(nbins = num_ticks)  # Set the locator
    # cbar.update_ticks()  # Update the ticks on the color bar
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f')) 

    #ax3.plot(a_mesh[arg_min_delta_ae], b_mesh[arg_min_delta_ae], 'ro', alpha = 0.2)

    plt.savefig('ae_delta_nfp_{}_r_{}_omn_{}_omt_{}.png'.format(nfp, r, omn, omt), dpi = 300, bbox_inches='tight')

    plt.show()
    
    # important for doing plots

    idx = arg_min_delta_ae

    print(idx)

    print(a_mesh[idx])

    print(b_mesh[idx])

    print(a_z_arr[idx])

    print(b_z_arr[idx])

    print(eta_arr[idx])

    print(B2c_arr[idx])

    print(rcrit_arr[idx])

    print(delB20_arr[idx])

    ae_computations(idx)

    stel = Qsc(rc=[1, a_mesh[idx], b_mesh[idx]], zs=[0.0, a_z_arr[idx], b_z_arr[idx]], \
        nfp=nfp, etabar=eta_arr[idx]+0.3, B0=B0, order = "r1") #, B2c = B2c_arr[idx], nphi=nphi)
    stel.spsi = 1
    stel.calculate()
    #print(stel.r_singularity)
    print(stel.iotaN)
    plot_boundary_jaime(stel, fieldlines=False, r = 0.015)
    stel.plot_boundary(r = r, fieldlines=True)#, filename='2nd_ae_config_nfp_{}_r_{}_omn_{}_omt_{}_fieldlines.png'.format(nfp, r, omn, omt))

    stel = Qsc(rc=[1, a_mesh[idx], b_mesh[idx]], zs=[0.0, a_z_arr[idx], b_z_arr[idx]], \
        nfp=nfp, etabar=eta_arr[idx]-0.3, B0=B0, order = "r1") #, B2c = B2c_arr[idx], nphi=nphi)
    stel.spsi = 1
    stel.calculate()
    #print(stel.r_singularity)
    print(stel.iotaN)
    plot_boundary_jaime(stel, fieldlines=False, r = 0.015)
    stel.plot_boundary(r = r, fieldlines=True)#, filename='2nd_ae_config_nfp_{}_r_{}_omn_{}_omt_{}_fieldlines.png'.format(nfp, r, omn, omt))

    stel = Qsc(rc=[1, a_mesh[idx], b_mesh[idx]], zs=[0.0, a_z_arr[idx], b_z_arr[idx]], \
        nfp=nfp, etabar=eta_arr[idx], B0=B0, order = "r2", B2c = B2c_arr[idx]*-1, nphi=nphi)
    stel.spsi = 1
    stel.calculate()
    print(stel.r_singularity)
    print(stel.iotaN)
    plot_boundary_jaime(stel, fieldlines=False, r = 0.015)
    stel.plot_boundary(r = r, fieldlines=True)#, filename='2nd_ae_config_nfp_{}_r_{}_omn_{}_omt_{}_fieldlines.png'.format(nfp, r, omn, omt))


