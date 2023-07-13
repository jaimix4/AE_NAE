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
import pandas as pd
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


# def ae_computations(idx):

#     # if rcrit_arr[idx] < 1e-4 or rcrit_arr[idx] > 1e5:

#     #     return (np.nan, np.nan)

#     # idx = (idx_b, idx_a)

#     stel = Qsc(rc=[1, a_mesh[idx], b_mesh[idx]], zs=[0.0, a_z_arr[idx], b_z_arr[idx]], \
#         nfp=nfp, etabar=eta_arr[idx], B0=B0, order = "r2", B2c = B2c_arr[idx], nphi=nphi)
#     stel.spsi = 1
#     stel.calculate()
#     alpha = 1.0
#     stel.r = r 
#     try:
#         NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=alpha, N_turns=3, nphi=nphi,
#                                 lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
#         NAE_AE.calc_AE(omn=stel.spsi*omn,omt=stel.spsi*omt,omnigenous=omnigenous)

#         ae_second = NAE_AE.ae_tot
#     except:
#         print('could not compute AE')
#         ae_second = np.nan

#     ae_first = ae_nor_nae(stel, r, lam_res, 1, 1, omn, omt, plot = False)[-1]

#     return (ae_first, ae_second)

# ####################
# # ae computations parameters

eps = np.finfo(float).eps
    
#@njit
def fourier_interpolation(fk, x):
    """
    Interpolate data that is known on a uniform grid in [0, 2pi).

    This routine is based on the
    matlab routine fourint.m in the DMSuite package by S.C. Reddy and J.A.C. Weideman, available at
    http://www.mathworks.com/matlabcentral/fileexchange/29
    or here:
    http://dip.sun.ac.za/~weideman/research/differ.html  

    fk:  Vector of y-coordinates of data, at equidistant points
         x(k) = (k-1)*2*pi/N,  k = 1...N
    x:   Vector of x-values where interpolant is to be evaluated.
    output: Vector of interpolated values.
    """

    N = len(fk)
    M = len(x)

    # Compute equidistant points
    #xk = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    xk = (np.arange(N) * 2 * np.pi) / N

    # Weights for trig interpolation
    w = (-1.0) ** np.arange(0, N)
    #w = np.array((-1) ** np.arange(0, N), dtype='f')

    """
    x2 = x / 2
    xk2 = xk / 2

    # Compute quantities x - x(k)
    xk2_2D, x2_2D = np.meshgrid(xk2, x2)
    Dold = x2_2D - xk2_2D
    D = 0.5 * (np.outer(x, np.ones(N)) - np.outer(np.ones(M), xk))
    print(Dold - D)
    """
    D = 0.5 * (np.outer(x, np.ones(N)) - np.outer(np.ones(M), xk))
    
    if np.mod(N, 2) == 0:
        # Formula for N even
        D = 1 / np.tan(D + eps * (D==0))
    else:
        # Formula for N odd
        D = 1 / np.sin(D + eps * (D==0))

    # Evaluate interpolant as matrix-vector products
    #return np.matmul(D, w * fk) / np.matmul(D, w)
    return np.dot(D, w * fk) / np.dot(D, w)
    #return (D @ w * fk) / (D @ w)
    #return D.dot(w * fk) / D.dot(w)


def calculate_triangularity(stel, both = True, rotate = False):
    # This is set to work for stellarator symmetric configurations

    if stel.lasym:
        raise ValueError('Input nae is not stellarator symmetric!')
    def eval_at_middle(y):
        # Interpolate to the bottom of the well (to take into account the shift of the phi grid)
        pos = [np.pi]
        res = fourier_interpolation(y, pos)
        return res[0]
    # If rotate = True, rotate the magnetic axis
    if rotate:
        rc = stel.rc
        rc = [rc[i]*(-1)**i for i in range(len(rc))]
        stel.rc = rc
        zs = stel.zs
        zs = [zs[i]*(-1)**i for i in range(len(zs))]
        stel.zs = zs
        stel.calculate()

    # Calculate triangularity in the Frenet-Serret frame delta_X in paper)
    if stel.order == 'r1':
        triang = ['nan']
        raise ValueError('Input nae is only order r1, and hence has no triangularity!')
    else:
        if both:
            triang = -2*np.sign(stel.etabar)*(stel.X2c/stel.X1c-stel.Y2s/stel.Y1s)
            triang_nae = np.array([0.0, 0.0])
            triang_nae[0] = triang[0]
            triang_nae[1] = eval_at_middle(triang)
        else:
            triang_nae = -2*np.sign(stel.etabar)*(stel.X2c[0]/stel.X1c[0]-stel.Y2s[0]/stel.Y1s[0])

    # Calculate triangularity (projection) in the lab frame (modified form of Appendix C1, to include all the 
    # necessary terms; original expression dropped some terms `by symmetry' that do actually not)
    if both:
        tTh = stel.binormal_cylindrical[:,1]
        tZ = stel.binormal_cylindrical[:,2]
        kR = stel.normal_cylindrical[:,0]
        X1c = stel.X1c
        Y1s = stel.Y1s
        d_Y1c = np.matmul(stel.d_d_phi, stel.Y1c)
        R0 = stel.R0
        dd_R0 = stel.R0pp
        ddd_Z0 = stel.Z0ppp
        term_1 = tTh/R0/X1c*(0.5*Y1s*Y1s*(1+dd_R0/R0) - X1c*X1c + 2*(X1c*X1c - Y1s*Y1s)*dd_R0/(R0-dd_R0))
        term_2 = np.sign(tZ)*d_Y1c/R0
        term_3 = tZ*(X1c*X1c-Y1s*Y1s)*ddd_Z0/R0/X1c/(R0-dd_R0)
        triang_geo_temp = np.sign(stel.etabar)*kR*tTh*(term_1 + term_2 + term_3)
        triang_geo = np.array([0.0, 0.0])
        triang_geo[0] = triang_geo_temp[0]
        triang_geo[1] = eval_at_middle(triang_geo_temp)
    else:
        tTh = stel.binormal_cylindrical[0,1]
        tZ = stel.binormal_cylindrical[0,2]
        kR = stel.normal_cylindrical[0,0]
        X1c = stel.X1c[0]
        Y1s = stel.Y1s[0]
        d_Y1c = np.matmul(stel.d_d_phi, stel.Y1c)[0]
        R0 = stel.R0[0]
        dd_R0 = stel.R0pp[0]
        ddd_Z0 = stel.Z0ppp[0]
        term_1 = tTh/R0/X1c*(0.5*Y1s*Y1s*(1+dd_R0/R0) - X1c*X1c + 2*(X1c*X1c - Y1s*Y1s)*dd_R0/(R0-dd_R0))
        term_2 = np.sign(tZ)*d_Y1c/R0
        term_3 = tZ*(X1c*X1c-Y1s*Y1s)*ddd_Z0/R0/X1c/(R0-dd_R0)
        triang_geo_temp = np.sign(stel.etabar)*kR*tTh*(term_1 + term_2 + term_3)

    # Compute vertical elongation 
    if both:
        a = stel.curvature**2/stel.etabar**2
        elon_out = np.array([0.0, 0.0])
        elon = a
        elon_out[0] = elon[0]
        elon_out[1] = eval_at_middle(elon)
    else:
        elon_out = stel.curvature[0]**2/stel.etabar**2

    return triang_nae, triang_geo, triang_nae+triang_geo, elon_out





nphi = int(1e3+1)
lam_res = 2001
omn = 3
omt = 0
omnigenous = False

r = 0.01

# ok I have to do r = 0.02

####################

N = 150
# nfp = 3
B0 = 1

# # NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
# a_arr = np.linspace(0.0001, 0.3, N)
# b_arr = np.linspace(-0.06, 0.06, N)

# # mesh grid a_arr and b_arr
# a_mesh, b_mesh = np.meshgrid(a_arr, b_arr)

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


nfp_arr = [2, 3, 4]


df_qa_list = []

df_qh_list = []



qa_eta_list = []
qa_b2c_list = []
qa_deltae_list = []

qh_eta_list = []
qh_b2c_list = []
qh_deltae_list = []


fig = plt.figure(figsize=(22/1.5, 7.4/1.5))
grid = gridspec.GridSpec(2, 3, wspace=0.3, hspace = 0.54, height_ratios=[1, 0.05])


for idx_plot, nfp in enumerate(nfp_arr):


    ax = plt.subplot(grid[0, idx_plot])


    a_mesh = np.load('shapes_2nd/a_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    b_mesh = np.load('shapes_2nd/b_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    a_z_arr = np.load('shapes_2nd/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    b_z_arr = np.load('shapes_2nd/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    eta_arr = np.load('shapes_2nd/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    B2c_arr = np.load('shapes_2nd/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    delB20_arr = np.load('shapes_2nd/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    rcrit_arr = np.load('shapes_2nd/rcrit_arr_N_{}_nfp_{}.npy'.format(N, nfp))

    ae_first_arr = np.load('ae_shapes_2nd/ae_first_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))
    ae_second_arr = np.load('ae_shapes_2nd/ae_second_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))


    nae_tria_0_arr = np.load('ae_shapes_2nd/nae_tria_0_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))
    nae_tria_middle_arr = np.load('ae_shapes_2nd/nae_tria_middle_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))

    geo_tria_0_arr = np.load('ae_shapes_2nd/geo_tria_0_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))
    geo_tria_middle_arr = np.load('ae_shapes_2nd/geo_tria_middle_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))

    total_tria_0_arr = np.load('ae_shapes_2nd/total_tria_0_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))
    total_tria_middle_arr = np.load('ae_shapes_2nd/total_tria_middle_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))

    elong_0_arr = np.load('ae_shapes_2nd/elong_0_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))
    elong_middle_arr = np.load('ae_shapes_2nd/elong_middle_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))

    iotaN_arr = np.load('ae_shapes_2nd/iotaN_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))




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


    ####################


    # make negatives values of ae_second_arr to be nan
    ae_second_arr[ae_second_arr < 0] = np.nan

    # getting rid of values which r_crit is smaller than r
    # since they are "not valid for that ordering"
    ae_first_arr[rcrit_arr < r] = np.nan
    ae_second_arr[rcrit_arr < r] = np.nan

    # make nan values of ae_second_arr to also be nan on ae_first_arr

    ae_first_arr[np.isnan(ae_second_arr)] = np.nan


    delta_ae = (ae_second_arr/ae_first_arr - 1)
    delta_ae[np.isnan(ae_second_arr)] = np.nan
    delta_ae[rcrit_arr < r] = np.nan


    delta_ae[delta_ae < -0.6] = np.nan
    delta_ae[delta_ae > 0.6] = np.nan

    # delta_ae[delta_ae < -25.0] = np.nan
    # delta_ae[delta_ae > 25.0] = np.nan

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

    # delta_ae[delta_ae > 0] = np.nan
    # delta_ae[delta_ae < -0.3] = np.nan

    ############ plot the whole thing #####################

    levels = np.linspace(-55.0, 55.0, 300)
    delta_plot = ax.contourf(a_mesh, b_mesh, delta_ae/r, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')

    ax.set_title(r'$\delta\hat{A}/r \quad r = $' + ' {}'.format(r), fontsize = 24)
    ax.set_xlabel(r'$a$', fontsize = 28)
    

    # ad text to the plot
    ax.text(0.02, 0.03, r'$N_{\rm fp} = $' + ' {}'.format(nfp), transform=ax.transAxes, fontsize=15)

    # put y ticks on right size

    if nfp == 4:

        ax.set_ylabel(r'$b$', fontsize = 28)
        ax.yaxis.set_label_position("right")

        cax = fig.add_subplot(grid[1, :])  # Colorbar subplot for plots 1 and 2
        cbar = fig.colorbar(delta_plot, cax=cax, orientation='horizontal')
        cbar.ax.locator_params(nbins=5)

        #plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

        # Set the number of ticks for the color bar
        # num_ticks = 5  # Set the desired number of ticks
        # cbar.locator = ticker.MaxNLocator(nbins = num_ticks)  # Set the locator
        # cbar.update_ticks()  # Update the ticks on the color bar
        cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) 


    ######### adding lines #########

    #ax = ax3

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


    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


    ################## QA case ##################


    delta_ae = (ae_second_arr/ae_first_arr - 1)
    delta_ae[np.isnan(ae_second_arr)] = np.nan
    delta_ae[rcrit_arr < r] = np.nan

    delta_ae[delta_ae < -0.6] = np.nan
    delta_ae[delta_ae > 0.6] = np.nan

    # delta_ae[delta_ae < -25.0] = np.nan
    # delta_ae[delta_ae > 25.0] = np.nan

    if nfp == 3:

        delta_ae[delta_ae > 0] = np.nan
        delta_ae[a_mesh > 0.13] = np.nan
        delta_ae[b_mesh > 0.025] = np.nan
        delta_ae[b_mesh < -0.025] = np.nan

    elif nfp == 4:

        delta_ae[delta_ae > 0] = np.nan
        delta_ae[a_mesh > 0.075] = np.nan

    elif nfp == 2:

        delta_ae[delta_ae > 0] = np.nan
        delta_ae[a_mesh > 0.24] = np.nan
        delta_ae[a_mesh < 0.08] = np.nan
        delta_ae[b_mesh < -0.03] = np.nan
        delta_ae[b_mesh > 0.03] = np.nan

    delta_ae = delta_ae/r

    idx_nonan_delta_ae = np.argwhere(~np.isnan(delta_ae))#[::-1]

    idx_nonan_delta_ae = idx_nonan_delta_ae[np.argsort(delta_ae[idx_nonan_delta_ae[:, 0], idx_nonan_delta_ae[:, 1]])]

    # array with 50 random sample from the idx_nonan_delta_ae array

    # idx_nonan_delta_ae = idx_nonan_delta_ae[np.random.choice(idx_nonan_delta_ae.shape[0], 50, replace=False)]
    print(idx_nonan_delta_ae)
    print(idx_nonan_delta_ae.shape)

    ########################## eta, b2c ##########################

    qa_eta_list.append(eta_arr[idx_nonan_delta_ae[:, 0], idx_nonan_delta_ae[:, 1]])
    qa_b2c_list.append(B2c_arr[idx_nonan_delta_ae[:, 0], idx_nonan_delta_ae[:, 1]])
    qa_deltae_list.append(ae_second_arr[idx_nonan_delta_ae[:, 0], idx_nonan_delta_ae[:, 1]])


    #############################################################


    max_num_configs = len(idx_nonan_delta_ae)
    cap = 50

    idx_nonan_delta_ae = idx_nonan_delta_ae[np.random.choice(idx_nonan_delta_ae.shape[0], cap, replace=False)]

    skip = max_num_configs // cap

    corr_data = np.zeros((cap, 22))

    #header_corr_data = np.array(["a", "b", "helicity", "eta", "delB20", "total_tria", "elongation", "B2c", "r_crit", "delta_ae"])

    count_special = 0

    count_pos = 0

    for count, point in enumerate(idx_nonan_delta_ae):

        idx = (point[0], point[1])


        if nfp == 2:

            if count == 0:

                ax.plot(a_mesh[idx], b_mesh[idx], 'b^', alpha = 0.1, markersize = 4, label = "QA")

            ax.plot(a_mesh[idx], b_mesh[idx], 'b^', alpha = 0.1, markersize = 4)

        elif nfp == 3:

            if count == 0:

                ax.plot(a_mesh[idx], b_mesh[idx], 'g^', alpha = 0.1, markersize = 4, label = "QA")

            ax.plot(a_mesh[idx], b_mesh[idx], 'g^', alpha = 0.1, markersize = 4)

        elif nfp == 4:

            if count == 0:

                ax.plot(a_mesh[idx], b_mesh[idx], 'r^', alpha = 0.1, markersize = 4, label = "QA")

            ax.plot(a_mesh[idx], b_mesh[idx], 'r^', alpha = 0.1, markersize = 4)



        stel = Qsc(rc=[1, a_mesh[idx], b_mesh[idx]], zs=[0.0, a_z_arr[idx], b_z_arr[idx]], \
        nfp=nfp, etabar=eta_arr[idx], B0=B0, order = "r2", B2c = B2c_arr[idx], nphi=nphi)

        #t_nae, t_geo, t_tot, elon = calculate_triangularity(stel, both = True)


        t_nae = [nae_tria_0_arr[idx], nae_tria_middle_arr[idx]]
        t_geo = [geo_tria_0_arr[idx], geo_tria_middle_arr[idx]]
        t_tot = [total_tria_0_arr[idx], total_tria_middle_arr[idx]]
        elon = [elong_0_arr[idx], elong_middle_arr[idx]]

        if t_nae[0] < -20:
            print("QA t_nae_0 < -20 for:")
            print(nfp)
            print(idx)
            print(t_nae)

        if t_nae[0] > 30:
            print("QA t_nae_0 > 30 for:")
            print(nfp)
            print(idx)
            print(t_nae)

        if delta_ae[idx] < -34:
            print("QA delta_ae < -38 for:")
            print(nfp)
            print(idx)
            print(delta_ae[idx])

        
        helicity  = stel.iotaN - stel.iota

        # index of maximum value on elon
        max_idx = np.argmax(elon)


        corr_data[count, :] = np.array([a_mesh[idx], b_mesh[idx], \
            helicity, stel.axis_length, eta_arr[idx], stel.iota, stel.iotaN, delB20_arr[idx], stel.G0, np.abs(stel.G0/stel.iotaN), \
            t_nae[0], t_nae[1], t_geo[0], t_geo[1], t_tot[0], t_tot[1], elon[0], elon[1], max(elon), \
            B2c_arr[idx], rcrit_arr[idx], delta_ae[idx]])

        # corr_data[count, :] = np.array([a_mesh[idx], b_mesh[idx], \
        #     helicity, stel.axis_length, eta_arr[idx], stel.iota, stel.iotaN, delB20_arr[idx], stel.G0, np.abs(stel.G0/stel.iotaN), \
        #     t_nae[0], t_nae[1], t_nae[max_idx], t_geo[1], t_tot[0], t_tot[1], elon[max_idx], elon[1], max(elon), \
        #     B2c_arr[idx], rcrit_arr[idx], delta_ae[idx]])


    df_qa_list.append(pd.DataFrame(corr_data, columns=["a", "b", "helicity", "axis_length", "eta", "iota", "iotaN", "delB20", "G0", "|G0/iotaN|", \
        "nae_tria_0", "nae_tria_middle", "geo_tria_0", "geo_tria_middle", "total_tria_0", "total_tria_middle", \
            "elon_0", "elon_middle", "max_elon", "B2c", "r_crit", "delta_ae"]).copy() )

    df = pd.DataFrame(corr_data, columns=["a", "b", "helicity", "axis_length", "eta", "iota", "iotaN", "delB20", "G0", "|G0/iotaN|", \
        "nae_tria_0", "nae_tria_middle", "geo_tria_0", "geo_tria_middle", "total_tria_0", "total_tria_middle", \
            "elon_0", "elon_middle", "max_elon", "B2c", "r_crit", "delta_ae"])



    print("########### QA correlation matrix ###########")
    print("nfp = ", nfp)



    ######## analysis QH ###########

    delta_ae = (ae_second_arr/ae_first_arr - 1)#/r
    delta_ae[np.isnan(ae_second_arr)] = np.nan
    delta_ae[rcrit_arr < r] = np.nan

    delta_ae[delta_ae < -0.6] = np.nan
    delta_ae[delta_ae > 0.6] = np.nan

    # delta_ae[delta_ae < -25.0] = np.nan
    # delta_ae[delta_ae > 25.0] = np.nan

    if nfp == 3:

        # delta_ae[delta_ae > 0] = np.nan
        delta_ae[a_mesh < 0.125] = np.nan


    elif nfp == 4:

        # delta_ae[delta_ae > 0] = np.nan
        delta_ae[a_mesh < 0.070] = np.nan

    elif nfp == 2:

        # delta_ae[delta_ae > 0] = np.nan
        delta_ae[a_mesh < 0.27] = np.nan

    delta_ae = delta_ae/r

    idx_nonan_delta_ae = np.argwhere(~np.isnan(delta_ae))#[::-1]

    idx_nonan_delta_ae = idx_nonan_delta_ae[np.argsort(delta_ae[idx_nonan_delta_ae[:, 0], idx_nonan_delta_ae[:, 1]])]

    ##################

    qh_eta_list.append(eta_arr[idx_nonan_delta_ae[:, 0], idx_nonan_delta_ae[:, 1]])
    qh_b2c_list.append(B2c_arr[idx_nonan_delta_ae[:, 0], idx_nonan_delta_ae[:, 1]])
    qh_deltae_list.append(ae_second_arr[idx_nonan_delta_ae[:, 0], idx_nonan_delta_ae[:, 1]])

    #################

    
    max_num_configs = len(idx_nonan_delta_ae)
    cap = cap
    idx_nonan_delta_ae = idx_nonan_delta_ae[np.random.choice(idx_nonan_delta_ae.shape[0], cap, replace=False)]

    skip = max_num_configs // cap

    corr_data = np.zeros((cap, 22))

    #header_corr_data = np.array(["a", "b", "helicity", "eta", "delB20", "total_tria", "elongation", "B2c", "r_crit", "delta_ae"])

    count_special = 0

    count_pos = 0

    count_neg = 0

    for count, point in enumerate(idx_nonan_delta_ae):

        idx = (point[0], point[1])


        if nfp == 2:

            if count == 0:

                ax.plot(a_mesh[idx], b_mesh[idx], 'bo', alpha = 0.15, markersize = 4, label = "QH")

            ax.plot(a_mesh[idx], b_mesh[idx], 'bo', alpha = 0.15, markersize = 4)

        elif nfp == 3:

            if count == 0:

                ax.plot(a_mesh[idx], b_mesh[idx], 'go', alpha = 0.15, markersize = 4, label = "QH")

            ax.plot(a_mesh[idx], b_mesh[idx], 'go', alpha = 0.15, markersize = 4)

        elif nfp == 4:

            if count == 0:

                ax.plot(a_mesh[idx], b_mesh[idx], 'ro', alpha = 0.15, markersize = 4, label = "QH")

            ax.plot(a_mesh[idx], b_mesh[idx], 'ro', alpha = 0.15, markersize = 4)

        stel = Qsc(rc=[1, a_mesh[idx], b_mesh[idx]], zs=[0.0, a_z_arr[idx], b_z_arr[idx]], \
            nfp=nfp, etabar=eta_arr[idx], B0=B0, order = "r2", B2c = B2c_arr[idx], nphi=nphi)
        stel.spsi = 1
        stel.calculate()

        # t_nae, t_geo, t_tot, elon = calculate_triangularity(stel, both = True)

        t_nae = [nae_tria_0_arr[idx], nae_tria_middle_arr[idx]]
        t_geo = [geo_tria_0_arr[idx], geo_tria_middle_arr[idx]]
        t_tot = [total_tria_0_arr[idx], total_tria_middle_arr[idx]]
        elon = [elong_0_arr[idx], elong_middle_arr[idx]]

        if t_nae[0] < -15:
            print("QH t_nae_0 < -15 for:")
            print(nfp)
            print(idx)
            print(t_nae)

        if t_nae[0] > 10:
            print("QH t_nae_0 > 10 for:")
            print(nfp)
            print(idx)
            print(t_nae)

        if delta_ae[idx] < -34:
            print("QH delta_ae < -34 for:")
            print(nfp)
            print(idx)
            print(delta_ae[idx])
        

        helicity  = stel.iotaN - stel.iota

        max_idx = np.argmax(elon)

        corr_data[count, :] = np.array([a_mesh[idx], b_mesh[idx], \
            helicity, stel.axis_length, eta_arr[idx], stel.iota, stel.iotaN, delB20_arr[idx], stel.G0, np.abs(stel.G0/stel.iotaN), \
            t_nae[0], t_nae[1], t_geo[0], t_geo[1], t_tot[0], t_tot[1], elon[0], elon[1], max(elon), \
            B2c_arr[idx], rcrit_arr[idx], delta_ae[idx]])

        # corr_data[count, :] = np.array([a_mesh[idx], b_mesh[idx], \
        #     helicity, stel.axis_length, eta_arr[idx], stel.iota, stel.iotaN, delB20_arr[idx], stel.G0, np.abs(stel.G0/stel.iotaN), \
        #     t_nae[0], t_nae[1], t_nae[max_idx], t_geo[1], t_tot[0], t_tot[1], elon[max_idx], elon[1], max(elon), \
        #     B2c_arr[idx], rcrit_arr[idx], delta_ae[idx]])

    df = pd.DataFrame(corr_data, columns=["a", "b", "helicity", "axis_length", "eta", "iota", "iotaN", "delB20", "G0", "|G0/iotaN|", \
        "nae_tria_0", "nae_tria_middle", "geo_tria_0", "geo_tria_middle", "total_tria_0", "total_tria_middle", \
            "elon_0", "elon_middle", "max_elon", "B2c", "r_crit", "delta_ae"])


    df_qh_list.append(pd.DataFrame(corr_data, columns=["a", "b", "helicity", "axis_length", "eta", "iota", "iotaN", "delB20", "G0", "|G0/iotaN|", \
        "nae_tria_0", "nae_tria_middle", "geo_tria_0", "geo_tria_middle", "total_tria_0", "total_tria_middle", \
            "elon_0", "elon_middle", "max_elon", "B2c", "r_crit", "delta_ae"]).copy() )


    print("########### QH correlation matrix ###########")
    print("nfp = ", nfp)
    # print(df.corr()["delta_ae"])

    ax.legend(markerscale = 3., ncols = 2, loc = "upper center")



plt.savefig("samples_geo_nfp_{}_r_{}.png".format(nfp, r), dpi = 300, bbox_inches = "tight")
plt.show()



df_qa = pd.concat(df_qa_list, ignore_index=True)

# print(df_qa)

# print(df_qa.corr()["delta_ae"])

df_qh = pd.concat(df_qh_list, ignore_index=True)

# print(df_qh)


# print(df_qh.corr()["delta_ae"])


# here normal input parameters plots 

fig = plt.figure(figsize=(22/1.5, 11/1.5))
grid = gridspec.GridSpec(2, 4, wspace=0.3, hspace = 0.4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

size_marker = 10

columns=["a", "b", "helicity", "axis_length", "eta", "iota", "iotaN", "delB20", "G0", "|G0/iotaN|", \
        "nae_tria_0", "nae_tria_middle", "geo_tria_0", "geo_tria_middle", "total_tria_0", "total_tria_middle", \
            "elon_0", "elon_middle", "max_elon", "B2c", "r_crit", "delta_ae"]


columns_first_graph_qa = ["eta", "iota", "G0", "|G0/iotaN|", "B2c", "delB20", "r_crit", "elon_0"]

columns_first_graph_qa_latex = [r'$\bar{\eta}$', r'$\iota_{0}$', r'$G_0$', r'$|G_0/\bar{\iota}|$', r'$B_{2c}$', r'$\Delta B_{20}$', r'$r_{\rm crit.}$', r'$\kappa_{\phi = 0}^{\rm nae}$']

for num, x_data in enumerate(columns_first_graph_qa):

    if num < 4:

        ax = plt.subplot(grid[0, num])

        ax.scatter(df_qa[x_data].to_numpy()[:cap], df_qa["delta_ae"].to_numpy()[:cap], s = size_marker,  c = "blue", marker = '^', label = "QA " + r'$N_{\rm fp} = 2$')
        ax.scatter(df_qa[x_data].to_numpy()[cap:2*cap], df_qa["delta_ae"].to_numpy()[cap:2*cap], s = size_marker,  c = "green", marker = '^', label = "QA " + r'$N_{\rm fp} = 3$')
        ax.scatter(df_qa[x_data].to_numpy()[2*cap:], df_qa["delta_ae"].to_numpy()[2*cap:], s = size_marker,  c = "red", marker = '^', label = "QA " + r'$N_{\rm fp} = 4$')

    else:

        ax = plt.subplot(grid[1, num - 4])

        ax.scatter(df_qa[x_data].to_numpy()[:cap], df_qa["delta_ae"].to_numpy()[:cap], s = size_marker, c = "blue", marker = '^', label = "QA " + r'$N_{\rm fp} = 2$')
        ax.scatter(df_qa[x_data].to_numpy()[cap:2*cap], df_qa["delta_ae"].to_numpy()[cap:2*cap], s = size_marker,  c = "green", marker = '^', label = "QA " + r'$N_{\rm fp} = 3$')
        ax.scatter(df_qa[x_data].to_numpy()[2*cap:], df_qa["delta_ae"].to_numpy()[2*cap:], s = size_marker, c = "red", marker = '^', label = "QA " + r'$N_{\rm fp} = 4$')

    ax.set_xlabel(columns_first_graph_qa_latex[num], fontsize = 24)

    if num == 0 or num == 4:

        ax.set_ylabel(r"$\delta \hat{A}/r$", fontsize = 24)

    # put legend with no repeate labels
    
    if x_data == "delB20":

        ax.set_xscale("log")

    if num == 3:

        ax.legend(ncol = 3, markerscale=3., loc = "upper center", bbox_to_anchor=(-1.36, 1.4), fontsize = 22)

plt.savefig("deltaae_geo_first_QA_r_{}.png".format(r), dpi = 300, bbox_inches = "tight")
plt.show()



############################################################################


size_marker = 8

fig = plt.figure(figsize=(22/1.5, 11/1.5))
grid = gridspec.GridSpec(2, 4, wspace=0.3, hspace = 0.4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])


columns_second_graph_qh = ["eta", "iota", "G0", "|G0/iotaN|", "B2c", "delB20", "r_crit", "elon_middle"]

columns_second_graph_qh_latex = [r'$\bar{\eta}$', r'$\iota_{0}$', r'$G_0$', r'$|G_0/\bar{\iota}|$', r'$B_{2c}$', r'$\Delta B_{20}$', r'$r_{\rm crit.}$', r'$\kappa_{\phi = \pi / N_{\rm fp}}^{\rm nae}$']



for num, x_data in enumerate(columns_second_graph_qh):

    if num < 4:

        ax = plt.subplot(grid[0, num])

        ax.scatter(df_qh[x_data].to_numpy()[:cap], df_qh["delta_ae"].to_numpy()[:cap], s = size_marker, c = "blue", marker = 'o', label = "QH " + r'$N_{\rm fp} = 2$')
        ax.scatter(df_qh[x_data].to_numpy()[cap:2*cap], df_qh["delta_ae"].to_numpy()[cap:2*cap], s = size_marker, c = "green", marker = 'o', label = "QH " + r'$N_{\rm fp} = 3$')
        ax.scatter(df_qh[x_data].to_numpy()[2*cap:], df_qh["delta_ae"].to_numpy()[2*cap:], s = size_marker, c = "red", marker = 'o', label = "QH " + r'$N_{\rm fp} = 4$')

    else:

        ax = plt.subplot(grid[1, num - 4])

        ax.scatter(df_qh[x_data].to_numpy()[:cap], df_qh["delta_ae"].to_numpy()[:cap], s  = size_marker,  c = "blue", marker = 'o', label = "QH " + r'$N_{\rm fp} = 2$')
        ax.scatter(df_qh[x_data].to_numpy()[cap:2*cap], df_qh["delta_ae"].to_numpy()[cap:2*cap], s = size_marker, c = "green", marker = 'o', label = "QH " + r'$N_{\rm fp} = 3$')
        ax.scatter(df_qh[x_data].to_numpy()[2*cap:], df_qh["delta_ae"].to_numpy()[2*cap:], s = size_marker, c = "red", marker = 'o', label = "QH " + r'$N_{\rm fp} = 4$')

    ax.set_xlabel(columns_second_graph_qh_latex[num], fontsize = 24)

    if num == 0 or num == 4:

        ax.set_ylabel(r"$\delta \hat{A}/r$")

    # put legend with no repeate labels
    
    if x_data == "delB20":

        ax.set_xscale("log")

    if num == 3:

        ax.legend(ncol = 3, markerscale=3., loc = "upper center", bbox_to_anchor=(-1.36, 1.4), fontsize = 22)



plt.savefig("delta_ae_geo_second_QH_r_{}.png".format(r), dpi = 300, bbox_inches = "tight")
plt.show()


# trinagularity plots for QA


fig = plt.figure(figsize=(18/1.5, 18/1.5))

grid = gridspec.GridSpec(2, 2, wspace=0.2, hspace = 0.4, height_ratios=[1, 1], width_ratios=[1, 1])

size_marker = 14

ax1 = plt.subplot(grid[0, 0])

ax1.scatter(df_qa["nae_tria_0"].to_numpy()[:cap], df_qa["delta_ae"].to_numpy()[:cap], s = size_marker*1.5, c = "blue", marker = '^', label = "QA " + r'$N_{\rm fp} = 2$')
ax1.scatter(df_qa["nae_tria_0"].to_numpy()[cap:2*cap], df_qa["delta_ae"].to_numpy()[cap:2*cap], s = size_marker*1.5, c = "green", marker = '^', label = "QA " + r'$N_{\rm fp} = 3$')
ax1.scatter(df_qa["nae_tria_0"].to_numpy()[2*cap:], df_qa["delta_ae"].to_numpy()[2*cap:], s = size_marker*1.5, c = "red", marker = '^', label = "QA " + r'$N_{\rm fp} = 4$')

ax1.set_ylabel(r"$\delta \hat{A}/r$", fontsize = 24)
ax1.set_xlabel(r"$\delta_{\phi = 0}$", fontsize = 24)

# ax1.legend(ncol = 1, markerscale=3., fontsize = 20)

ax2 = plt.subplot(grid[0, 1])

ax2.scatter(df_qa["nae_tria_middle"].to_numpy()[:cap], df_qa["delta_ae"].to_numpy()[:cap], s = size_marker*1.5, c = "blue", marker = '^', label = "QA " + r'$N_{\rm fp} = 2$')
ax2.scatter(df_qa["nae_tria_middle"].to_numpy()[cap:2*cap], df_qa["delta_ae"].to_numpy()[cap:2*cap], s = size_marker*1.5, c = "green", marker = '^', label = "QA " + r'$N_{\rm fp} = 3$')
ax2.scatter(df_qa["nae_tria_middle"].to_numpy()[2*cap:], df_qa["delta_ae"].to_numpy()[2*cap:], s = size_marker*1.5, c = "red", marker = '^', label = "QA " + r'$N_{\rm fp} = 4$')

# ax2.set_ylabel(r"$\delta \hat{A}/r$")
ax2.set_xlabel(r"$\delta_{\phi = \pi / N_{\rm fp}}$", fontsize = 24)

ax2.legend(ncol = 3, markerscale=3., fontsize = 20, loc = "upper center", bbox_to_anchor=(-0.23, 1.2))

ax3 = plt.subplot(grid[1, 0])

ax3.scatter(df_qh["nae_tria_0"].to_numpy()[:cap], df_qh["delta_ae"].to_numpy()[:cap], s = size_marker, c = "blue", marker = 'o', label = "QH " + r'$N_{\rm fp} = 2$')
ax3.scatter(df_qh["nae_tria_0"].to_numpy()[cap:2*cap], df_qh["delta_ae"].to_numpy()[cap:2*cap], s = size_marker, c = "green", marker = 'o', label = "QH " + r'$N_{\rm fp} = 3$')
ax3.scatter(df_qh["nae_tria_0"].to_numpy()[2*cap:], df_qh["delta_ae"].to_numpy()[2*cap:], s = size_marker, c = "red", marker = 'o', label = "QH " + r'$N_{\rm fp} = 4$')

ax3.set_ylabel(r"$\delta \hat{A}/r$", fontsize = 24)
ax3.set_xlabel(r"$\delta_{\phi = 0}^{\rm nae}$", fontsize = 24)

# ax3.legend(ncol = 1, markerscale=3., fontsize = 20)

ax4 = plt.subplot(grid[1, 1])

ax4.scatter(df_qh["nae_tria_middle"].to_numpy()[:cap], df_qh["delta_ae"].to_numpy()[:cap], s = size_marker, c = "blue", marker = 'o', label = "QH " + r'$N_{\rm fp} = 2$')
ax4.scatter(df_qh["nae_tria_middle"].to_numpy()[cap:2*cap], df_qh["delta_ae"].to_numpy()[cap:2*cap], s = size_marker, c = "green", marker = 'o', label = "QH " + r'$N_{\rm fp} = 3$')
ax4.scatter(df_qh["nae_tria_middle"].to_numpy()[2*cap:], df_qh["delta_ae"].to_numpy()[2*cap:], s = size_marker, c = "red", marker = 'o', label = "QH " + r'$N_{\rm fp} = 4$')

# ax4.set_ylabel(r"$\delta \hat{A}/r$")

ax4.set_xlabel(r"$\delta_{\phi = \pi / N_{\rm fp}}^{\rm nae}$", fontsize = 24)

ax4.legend(ncol = 3, markerscale=3., fontsize = 20, loc = "upper center", bbox_to_anchor=(-0.23, 1.2))

plt.savefig("triangularity_plot_QA_QH_r_{}.png".format(r), dpi = 300, bbox_inches = "tight")
plt.show()


############################### 2D plot ######################################

fig = plt.figure(figsize=(22/1.5, 7.4/1.5))
grid = gridspec.GridSpec(2, 3, wspace=0.3, hspace = 0.54, height_ratios=[1, 0.05])

ax1 = plt.subplot(grid[0, 0])
ax2 = plt.subplot(grid[0, 1])
ax3 = plt.subplot(grid[0, 2])

levels = np.linspace(0.006, 0.04, 300)

ax1.tricontourf(qa_eta_list[0], qa_b2c_list[0], qa_deltae_list[0], levels = levels, cmap = mpl.colormaps['jet'], extend = "both")

ax2.tricontourf(qa_eta_list[1], qa_b2c_list[1], qa_deltae_list[1], levels = levels, cmap = mpl.colormaps['jet'], extend = "both")

ax3_plot = ax3.tricontourf(qa_eta_list[2], qa_b2c_list[2], qa_deltae_list[2], levels = levels, cmap = mpl.colormaps['jet'], extend = "both")


cax = fig.add_subplot(grid[1, :])  # Colorbar subplot for plots 1 and 2
cbar = fig.colorbar(ax3_plot, cax=cax, orientation='horizontal')
cbar.ax.locator_params(nbins=5)

        #plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

        # Set the number of ticks for the color bar
        # num_ticks = 5  # Set the desired number of ticks
        # cbar.locator = ticker.MaxNLocator(nbins = num_ticks)  # Set the locator
        # cbar.update_ticks()  # Update the ticks on the color bar
# cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) 
cbar.ax.ticklabel_format(style='sci', scilimits=(-2,-2), axis='both')

ax1.set_ylabel(r"$B_{2c}$", fontsize = 24)

ax1.set_xlabel(r"$\bar{\eta}$", fontsize = 24)
ax2.set_xlabel(r"$\bar{\eta}$", fontsize = 24)
ax3.set_xlabel(r"$\bar{\eta}$", fontsize = 24)

ax1.set_title(r"$\hat{A}$", fontsize = 24)
ax1.text(0.05, 0.9, "QA - " + r"$N_{\rm fp} = 2$", fontsize = 20, transform=ax1.transAxes)
ax2.set_title(r"$\hat{A}$", fontsize = 24)
ax2.text(0.05, 0.9, "QA - " + r"$N_{\rm fp} = 3$", fontsize = 20, transform=ax2.transAxes)
ax3.set_title(r"$\hat{A}$", fontsize = 24)
ax3.text(0.45, 0.9, "QA - " + r"$N_{\rm fp} = 4$", fontsize = 20, transform=ax3.transAxes)

plt.savefig("b2c_eta_ae_QA_r_{}.png".format(r), dpi = 300, bbox_inches = "tight")
plt.show()


############################### 2D plot ######################################

fig = plt.figure(figsize=(22/1.5, 7.4/1.5))
grid = gridspec.GridSpec(2, 3, wspace=0.3, hspace = 0.54, height_ratios=[1, 0.05])

ax1 = plt.subplot(grid[0, 0])
ax2 = plt.subplot(grid[0, 1])
ax3 = plt.subplot(grid[0, 2])

levels = np.linspace(0.006, 0.06, 300)

ax1.tricontourf(qh_eta_list[0], qh_b2c_list[0], qh_deltae_list[0], levels = levels, cmap = mpl.colormaps['jet'], extend = "both")

ax2.tricontourf(qh_eta_list[1], qh_b2c_list[1], qh_deltae_list[1], levels = levels, cmap = mpl.colormaps['jet'], extend = "both")

ax3_plot = ax3.tricontourf(qh_eta_list[2], qh_b2c_list[2], qh_deltae_list[2], levels = levels, cmap = mpl.colormaps['jet'], extend = "both")


cax = fig.add_subplot(grid[1, :])  # Colorbar subplot for plots 1 and 2
cbar = fig.colorbar(ax3_plot, cax=cax, orientation='horizontal')
cbar.ax.locator_params(nbins=5)

        #plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

        # Set the number of ticks for the color bar
        # num_ticks = 5  # Set the desired number of ticks
        # cbar.locator = ticker.MaxNLocator(nbins = num_ticks)  # Set the locator
        # cbar.update_ticks()  # Update the ticks on the color bar
# cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) 
# add scientific notation to color bar
cbar.ax.ticklabel_format(style='sci', scilimits=(-2,-2), axis='both')

ax1.set_ylabel(r"$B_{2c}$", fontsize = 24)

ax1.set_xlabel(r"$\bar{\eta}$", fontsize = 24)
ax2.set_xlabel(r"$\bar{\eta}$", fontsize = 24)
ax3.set_xlabel(r"$\bar{\eta}$", fontsize = 24)

ax1.set_title(r"$\hat{A}$", fontsize = 24)
ax1.text(0.05, 0.05, "QH - " + r"$N_{\rm fp} = 2$", fontsize = 20, transform=ax1.transAxes)
ax2.set_title(r"$\hat{A}$", fontsize = 24)
ax2.text(0.05, 0.05, "QH - " + r"$N_{\rm fp} = 3$", fontsize = 20, transform=ax2.transAxes)
ax3.set_title(r"$\hat{A}$", fontsize = 24)
ax3.text(0.05, 0.05, "QH - " + r"$N_{\rm fp} = 4$", fontsize = 20, transform=ax3.transAxes)

plt.savefig("b2c_eta_ae_QH_r_{}.png".format(r), dpi = 300, bbox_inches = "tight")
plt.show()

# for i in range(3):
#     plt.tricontourf(qa_eta_list[i], qa_b2c_list[i], qa_deltae_list[i], levels = 300, cmap = mpl.colormaps['jet'])
#     plt.xlabel(r"$\bar{\eta}$")
#     plt.ylabel(r"$B_{2c}$")
#     plt.colorbar()
#     plt.show()

# for i in range(3):
#     plt.tricontourf(qh_eta_list[i], qh_b2c_list[i], qh_deltae_list[i], levels = 300, cmap = mpl.colormaps['jet'])
#     plt.xlabel(r"$\bar{\eta}$")
#     plt.ylabel(r"$B_{2c}$")
#     plt.colorbar()
#     plt.show()



"""


delta_ae[delta_ae > 0] = np.nan
delta_ae[a_mesh < 0.15] = np.nan


# index of minimum value on delta_ae


idx_nonan_delta_ae = np.argwhere(~np.isnan(delta_ae))#[::-1]

idx_nonan_delta_ae = idx_nonan_delta_ae[np.argsort(delta_ae[idx_nonan_delta_ae[:, 0], idx_nonan_delta_ae[:, 1]])]

print(idx_nonan_delta_ae)

print(len(idx_nonan_delta_ae))

print(len(delta_ae.flatten()))
non_nan_delta_arr = delta_ae[~np.isnan(delta_ae)]

non_nan_delta_arr = np.sort(non_nan_delta_arr)

# reverse order of array
# non_nan_delta_arr = non_nan_delta_arr[::-1]

# print(non_nan_delta_arr)

# print(len(non_nan_delta_arr))


arg_min_delta_ae = np.nanargmin(delta_ae)

arg_min_delta_ae = np.unravel_index(np.nanargmin(delta_ae), delta_ae.shape)

print(arg_min_delta_ae)

print(delta_ae[arg_min_delta_ae])


fig = plt.figure()
levels = np.linspace(-20.0, 20.0, 300)
levels = np.linspace(-0.6, 0.6, 300)
# levels = 200
plt.contourf(a_mesh, b_mesh, delta_ae, levels = levels, cmap = mpl.colormaps['jet'])
plt.title('(AE2nd / AE1st - 1) ,  at r = {}, omn = {}, omt = {}'.format(r, omn, omt))
plt.colorbar()
plt.xlabel('a')
plt.ylabel('b')

plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

for count, point in enumerate(idx_nonan_delta_ae):

    #print(point)
    idx = (point[0], point[1])

    if count == 0:

        plt.plot(a_mesh[idx], b_mesh[idx], 'go', alpha = 0.05, markersize = 4)

    elif delta_ae[point[0], point[1]] < 0.0:

        plt.plot(a_mesh[idx], b_mesh[idx], 'ro', alpha = 0.05, markersize = 4)

    else:

        plt.plot(a_mesh[point[0], point[1]], b_mesh[point[0], point[1]], 'bo', alpha = 0.05, markersize = 4)

    if count == 600:

        break

plt.savefig('ae_second_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.png'.format(N, nfp, r, omn, omt), dpi = 300)

# plt.show()

# idx = arg_min_delta_ae

# stel = Qsc(rc=[1, a_mesh[idx], b_mesh[idx]], zs=[0.0, a_z_arr[idx], b_z_arr[idx]], \
#     nfp=nfp, etabar=eta_arr[idx], B0=B0, order = "r2", B2c = B2c_arr[idx], nphi=nphi)
# stel.spsi = 1
# stel.calculate()

# stel.plot_boundary(r = r)
######## eduardo plot#####

max_num_configs = len(idx_nonan_delta_ae)
cap = 100

skip = max_num_configs // cap

count_special = 0

B2c_edu_arr = np.zeros(max_num_configs)
eta_edu_arr = np.zeros(max_num_configs)
ae_edu_arr = np.zeros(max_num_configs)

# make array 1d of 100 elements with zero values



for count, point in enumerate(idx_nonan_delta_ae):

    idx = (point[0], point[1])

    B2c_edu_arr[count] = B2c_arr[idx]
    eta_edu_arr[count] = eta_arr[idx]
    ae_edu_arr[count] = ae_second_arr[idx]



fig = plt.figure()
# Plot the contour using the X, Y, and Z arrays
plt.tricontourf(eta_edu_arr,  B2c_edu_arr, ae_edu_arr, levels = 200, cmap = mpl.colormaps['jet'])
plt.colorbar()  # Add a colorbar

# Add labels and title
plt.xlabel('eta')
plt.ylabel('B2c')
plt.title('AE second at r = {}, omn = {}, omt = {}'.format(r, omn, omt))

plt.show()


##########################


# array with dimensions 8 x 500
max_num_configs = len(idx_nonan_delta_ae)
cap = 50

skip = max_num_configs // cap

corr_data = np.zeros((cap, 22))

#header_corr_data = np.array(["a", "b", "helicity", "eta", "delB20", "total_tria", "elongation", "B2c", "r_crit", "delta_ae"])


count_special = 0

for count, point in enumerate(idx_nonan_delta_ae):

    if count % skip == 0:
    
        idx = (point[0], point[1])

        plt.plot(a_mesh[idx], b_mesh[idx], 'go', alpha = 0.2, markersize = 4)

        stel = Qsc(rc=[1, a_mesh[idx], b_mesh[idx]], zs=[0.0, a_z_arr[idx], b_z_arr[idx]], \
        nfp=nfp, etabar=eta_arr[idx], B0=B0, order = "r2", B2c = B2c_arr[idx], nphi=nphi)

        t_nae, t_geo, t_tot, elon = calculate_triangularity(stel, both = True)

        print("--------------------")
        print("nae triangularity: 0, middle") 
        print(t_nae)
        print("nae geometry: 0, middle") 
        print(t_geo)
        print("total triangularity: 0, middle") 
        print(t_tot)
        print("elongation: 0, middle") 
        print(elon)
        print("--------------------")


        helicity  = stel.iotaN - stel.iota

        corr_data[count_special, :] = np.array([a_mesh[idx], b_mesh[idx], \
            helicity, stel.axis_length, eta_arr[idx], stel.iota, stel.iotaN, delB20_arr[idx], stel.G0, np.abs(stel.G0/stel.iotaN), \
            t_nae[0], t_nae[1], t_geo[0], t_geo[1], t_tot[0], t_tot[1], elon[0], elon[1], max(elon), \
            B2c_arr[idx], rcrit_arr[idx], delta_ae[idx]])

        count_special += 1

    if count_special == cap:

        break

df = pd.DataFrame(corr_data, columns=["a", "b", "helicity", "axis_length", "eta", "iota", "iotaN", "delB20", "G0", "|G0/iotaN|", \
    "nae_tria_0", "nae_tria_middle", "geo_tria_0", "geo_tria_middle", "total_tria_0", "total_tria_middle", \
        "elon_0", "elon_middle", "max_elon", "B2c", "r_crit", "delta_ae"])

correlation_pearson = df.corr()#["delta_ae"]
correlation_spearman = df.corr(method = "spearman")["delta_ae"]

fig = plt.figure()

plt.imshow(correlation_pearson, cmap = 'hot', interpolation = 'nearest')
print(df)

print(correlation_pearson)
print(correlation_spearman)

plt.show()


"""