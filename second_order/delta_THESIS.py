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
        NAE_AE.plot_AE_per_lam()
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

r = 0.0001
r = 0.001
r = 0.01


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

# ae_first_arr = np.zeros_like(a_mesh)
# ae_second_arr = np.zeros_like(a_mesh)

###### loading arrays of ae ##########

r = 0.0001
ae_first_arr_r_0001 = np.load('ae_shapes_2nd/ae_first_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))
ae_second_arr_r_0001 = np.load('ae_shapes_2nd/ae_second_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))

# make negatives values of ae_second_arr to be nan
ae_second_arr_r_0001[ae_second_arr_r_0001 < 0] = np.nan

# getting rid of values which r_crit is smaller than r
# since they are "not valid for that ordering"
ae_first_arr_r_0001[rcrit_arr < r] = np.nan
ae_second_arr_r_0001[rcrit_arr < r] = np.nan

# make nan values of ae_second_arr to also be nan on ae_first_arr

ae_first_arr_r_0001[np.isnan(ae_second_arr_r_0001)] = np.nan

delta_ae_r_0001 = (ae_second_arr_r_0001/ae_first_arr_r_0001 - 1)#/r
delta_ae_r_0001[rcrit_arr < r] = np.nan

delta_ae_r_0001[delta_ae_r_0001 < -0.6] = np.nan
delta_ae_r_0001[delta_ae_r_0001 > 0.6] = np.nan


r = 0.001
ae_first_arr_r_001 = np.load('ae_shapes_2nd/ae_first_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))
ae_second_arr_r_001 = np.load('ae_shapes_2nd/ae_second_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))

# make negatives values of ae_second_arr to be nan
ae_second_arr_r_001[ae_second_arr_r_001 < 0] = np.nan

# getting rid of values which r_crit is smaller than r
# since they are "not valid for that ordering"
ae_first_arr_r_001[rcrit_arr < r] = np.nan
ae_second_arr_r_001[rcrit_arr < r] = np.nan

# make nan values of ae_second_arr to also be nan on ae_first_arr

ae_first_arr_r_001[np.isnan(ae_second_arr_r_001)] = np.nan

delta_ae_r_001 = (ae_second_arr_r_001/ae_first_arr_r_001 - 1)#/r
delta_ae_r_001[rcrit_arr < r] = np.nan

delta_ae_r_001[delta_ae_r_001 < -0.6] = np.nan
delta_ae_r_001[delta_ae_r_001 > 0.6] = np.nan

r = 0.01
ae_first_arr_r_01 = np.load('ae_shapes_2nd/ae_first_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))
ae_second_arr_r_01 = np.load('ae_shapes_2nd/ae_second_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))

# make negatives values of ae_second_arr to be nan
ae_second_arr_r_01[ae_second_arr_r_01 < 0] = np.nan

# getting rid of values which r_crit is smaller than r
# since they are "not valid for that ordering"
ae_first_arr_r_01[rcrit_arr < r] = np.nan
ae_second_arr_r_01[rcrit_arr < r] = np.nan

# make nan values of ae_second_arr to also be nan on ae_first_arr

ae_first_arr_r_01[np.isnan(ae_second_arr_r_01)] = np.nan

delta_ae_r_01 = (ae_second_arr_r_01/ae_first_arr_r_01 - 1)#/r
delta_ae_r_01[rcrit_arr < r] = np.nan

delta_ae_r_01[delta_ae_r_01 < -0.6] = np.nan
delta_ae_r_01[delta_ae_r_01 > 0.6] = np.nan

r = 0.02
ae_first_arr_r_02 = np.load('ae_shapes_2nd/ae_first_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))
ae_second_arr_r_02 = np.load('ae_shapes_2nd/ae_second_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.npy'.format(N, nfp, r, omn, omt))

# make negatives values of ae_second_arr to be nan
ae_second_arr_r_02[ae_second_arr_r_02 < 0] = np.nan

# getting rid of values which r_crit is smaller than r
# since they are "not valid for that ordering"
ae_first_arr_r_02[rcrit_arr < r] = np.nan
ae_second_arr_r_02[rcrit_arr < r] = np.nan

# make nan values of ae_second_arr to also be nan on ae_first_arr

ae_first_arr_r_02[np.isnan(ae_second_arr_r_02)] = np.nan

delta_ae_r_02 = (ae_second_arr_r_02/ae_first_arr_r_02 - 1)#/r
delta_ae_r_02[rcrit_arr < r] = np.nan

delta_ae_r_02[delta_ae_r_02 < -0.6] = np.nan
delta_ae_r_02[delta_ae_r_02 > 0.6] = np.nan


#################### plotting r = 0.0001 ####################

fig = plt.figure(figsize=(21/1.5, 7/1.5))
#grid = gridspec.GridSpec(1, 3, wspace=0.3)

grid = gridspec.GridSpec(2, 3, height_ratios=[1, 0.05], wspace=0.3, hspace=0.55)

ax1 = plt.subplot(grid[0, 0])

ax2 = plt.subplot(grid[0, 1])

ax3 = plt.subplot(grid[0, 2])

#levels = np.linspace(np.nanmin(ae_first_arr), np.nanmax(ae_first_arr), 200)

# using max and minimum of ae_second_arr to make the levels

# if r == 0.0001:
#     levels = np.linspace(np.nanmin(ae_second_arr_r_0001), 0.0055, 200)
# else:
#     levels = np.linspace(np.nanmin(ae_second_arr_r_0001), np.nanmax(ae_second_arr_r_0001), 300)
# plot at first order
r = 0.0001

# delta_ae_r_0001[delta_ae_r_01 < 0.01] = np.nan

# put values of delta_ae_r_0001 to nan, where delta_ae_r_01 is nan
delta_ae_r_0001[np.isnan(delta_ae_r_01)] = np.nan

levels = np.linspace(-50, 50, 300)

ax1.contourf(a_mesh, b_mesh, delta_ae_r_0001/r, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
# plt.colorbar()
# ax1.set_title('AE first at r = {}, omn = {}, omt = {}'.format(r, omn, omt))
ax1.set_title(r'$\delta\hat{A} / r, \quad r = $' + ' {}'.format(r), fontsize = 28)
ax1.set_xlabel(r'$a$', fontsize = 28)
ax1.set_ylabel(r'$b$', fontsize = 28)

#plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

#fig = plt.figure()
# levels = np.linspace(np.nanmin(ae_second_arr), np.nanmax(ae_second_arr), 200)
r = 0.001

delta_ae_r_001[np.isnan(delta_ae_r_01)] = np.nan


ax2_plot = ax2.contourf(a_mesh, b_mesh, delta_ae_r_001/r, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
ax2.set_title(r'$\delta\hat{A} / r, \quad r = $' + ' {}'.format(r), fontsize = 28)
#ax2.set_xlabel(r'$s$', fontsize = 28)
ax2.set_xlabel(r'$a$', fontsize = 28)

# cax = fig.add_subplot(grid[1, :2])  # Colorbar subplot for plots 1 and 2
# cbar = fig.colorbar(ax2_plot, cax=cax, orientation='horizontal')
# cbar.ax.locator_params(nbins=8)

# # Set the number of ticks for the color bar
# # num_ticks = 6  # Set the desired number of ticks
# # cbar.locator = ticker.MaxNLocator(num_ticks)  # Set the locator
# # cbar.update_ticks()  # Update the ticks on the color bar
# cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f')) 
r = 0.01
# levels = np.linspace(-0.6, 0.6, 300)

#delta_ae_r_01[delta_ae_r_01 == np.nan] = np.nan


ax3_plot = ax3.contourf(a_mesh, b_mesh, delta_ae_r_01/r, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
ax3.set_title(r'$\delta\hat{A} / r, \quad r = $' + ' {}'.format(r), fontsize = 28)
#plt.title('(AE2nd / AE1st - 1) ,  at r = {}, omn = {}, omt = {}'.format(r, omn, omt))
#plt.colorbar()
ax3.set_xlabel(r'$a$', fontsize = 28)
#ax3.set_ylabel(r'$b$', fontsize = 28)

cax = fig.add_subplot(grid[1, :])  # Colorbar subplot for plots 1 and 2
cbar = fig.colorbar(ax3_plot, cax=cax, orientation='horizontal')
cbar.ax.locator_params(nbins=7)
#plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

# Set the number of ticks for the color bar
# num_ticks = 5  # Set the desired number of ticks
# cbar.locator = ticker.MaxNLocator(nbins = num_ticks)  # Set the locator
# cbar.update_ticks()  # Update the ticks on the color bar
cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f')) 

# ax3.plot(a_mesh[arg_min_delta_ae], b_mesh[arg_min_delta_ae], 'ro', alpha = 0.2)

plt.savefig('Delta_nfp_{}_omn_{}_omt_{}.png'.format(nfp, omn, omt), dpi = 300, bbox_inches='tight')

plt.show()

"""

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

fig = plt.figure(figsize=(21/1.5, 7/1.5))
#grid = gridspec.GridSpec(1, 3, wspace=0.3)

grid = gridspec.GridSpec(2, 3, height_ratios=[1, 0.05], wspace=0.3, hspace=0.55)

ax1 = plt.subplot(grid[0, 0])

ax2 = plt.subplot(grid[0, 1])

ax3 = plt.subplot(grid[0, 2])

#levels = np.linspace(np.nanmin(ae_first_arr), np.nanmax(ae_first_arr), 200)

# using max and minimum of ae_second_arr to make the levels

if r == 0.0001:
    levels = np.linspace(np.nanmin(ae_second_arr), 0.0055, 200)
else:
    levels = np.linspace(np.nanmin(ae_second_arr), np.nanmax(ae_second_arr), 300)
# plot at first order

ax1.contourf(a_mesh, b_mesh, ae_first_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
# plt.colorbar()
# ax1.set_title('AE first at r = {}, omn = {}, omt = {}'.format(r, omn, omt))
ax1.set_title(r'$\hat{A}_{L}$', fontsize = 28)
ax1.set_xlabel(r'$a$', fontsize = 28)
ax1.set_ylabel(r'$b$', fontsize = 28)

#plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

#fig = plt.figure()
# levels = np.linspace(np.nanmin(ae_second_arr), np.nanmax(ae_second_arr), 200)

ax2_plot = ax2.contourf(a_mesh, b_mesh, ae_second_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
ax2.set_title(r'$\hat{A}$', fontsize = 28)
#ax2.set_xlabel(r'$s$', fontsize = 28)
ax2.set_xlabel(r'$a$', fontsize = 28)

cax = fig.add_subplot(grid[1, :2])  # Colorbar subplot for plots 1 and 2
cbar = fig.colorbar(ax2_plot, cax=cax, orientation='horizontal')
cbar.ax.locator_params(nbins=8)

# Set the number of ticks for the color bar
# num_ticks = 6  # Set the desired number of ticks
# cbar.locator = ticker.MaxNLocator(num_ticks)  # Set the locator
# cbar.update_ticks()  # Update the ticks on the color bar
cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f')) 

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

delta_ae[delta_ae < -0.6] = np.nan
delta_ae[delta_ae > 0.6] = np.nan
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
#ax3.set_ylabel(r'$b$', fontsize = 28)

cax = fig.add_subplot(grid[1, 2])  # Colorbar subplot for plots 1 and 2
cbar = fig.colorbar(ax3_plot, cax=cax, orientation='horizontal')
cbar.ax.locator_params(nbins=5)
#plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

# Set the number of ticks for the color bar
# num_ticks = 5  # Set the desired number of ticks
# cbar.locator = ticker.MaxNLocator(nbins = num_ticks)  # Set the locator
# cbar.update_ticks()  # Update the ticks on the color bar
cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f')) 

ax3.plot(a_mesh[arg_min_delta_ae], b_mesh[arg_min_delta_ae], 'ro', alpha = 0.2)

plt.savefig('ae_delta_nfp_{}_r_{}_omn_{}_omt_{}.png'.format(nfp, r, omn, omt), dpi = 300, bbox_inches='tight')

plt.show()


idx = arg_min_delta_ae

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
    NAE_AE.plot_AE_per_lam()
    
except:
    print("jaime")

stel.plot_boundary(r = r)

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

    fig = plt.figure(figsize=(21/1.5, 7/1.5))
    #grid = gridspec.GridSpec(1, 3, wspace=0.3)

    grid = gridspec.GridSpec(2, 3, height_ratios=[1, 0.05], wspace=0.3, hspace=0.55)

    ax1 = plt.subplot(grid[0, 0])

    ax2 = plt.subplot(grid[0, 1])

    ax3 = plt.subplot(grid[0, 2])

    #levels = np.linspace(np.nanmin(ae_first_arr), np.nanmax(ae_first_arr), 200)

    # using max and minimum of ae_second_arr to make the levels
    
    if r == 0.0001:
        levels = np.linspace(np.nanmin(ae_second_arr), 0.0055, 200)
    else:
        levels = np.linspace(np.nanmin(ae_second_arr), np.nanmax(ae_second_arr), 300)
    # plot at first order

    ax1.contourf(a_mesh, b_mesh, ae_first_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
    # plt.colorbar()
    # ax1.set_title('AE first at r = {}, omn = {}, omt = {}'.format(r, omn, omt))
    ax1.set_title(r'$\hat{A}_{L}$', fontsize = 28)
    ax1.set_xlabel(r'$a$', fontsize = 28)
    ax1.set_ylabel(r'$b$', fontsize = 28)

    #plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

    #fig = plt.figure()
    # levels = np.linspace(np.nanmin(ae_second_arr), np.nanmax(ae_second_arr), 200)

    ax2_plot = ax2.contourf(a_mesh, b_mesh, ae_second_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
    ax2.set_title(r'$\hat{A}$', fontsize = 28)
    #ax2.set_xlabel(r'$s$', fontsize = 28)
    ax2.set_xlabel(r'$a$', fontsize = 28)

    cax = fig.add_subplot(grid[1, :2])  # Colorbar subplot for plots 1 and 2
    cbar = fig.colorbar(ax2_plot, cax=cax, orientation='horizontal')
    cbar.ax.locator_params(nbins=8)

    # Set the number of ticks for the color bar
    # num_ticks = 6  # Set the desired number of ticks
    # cbar.locator = ticker.MaxNLocator(num_ticks)  # Set the locator
    # cbar.update_ticks()  # Update the ticks on the color bar
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f')) 

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

    delta_ae[delta_ae < -0.6] = np.nan
    delta_ae[delta_ae > 0.6] = np.nan
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
    #ax3.set_ylabel(r'$b$', fontsize = 28)

    cax = fig.add_subplot(grid[1, 2])  # Colorbar subplot for plots 1 and 2
    cbar = fig.colorbar(ax3_plot, cax=cax, orientation='horizontal')
    cbar.ax.locator_params(nbins=5)
    #plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

    # Set the number of ticks for the color bar
    # num_ticks = 5  # Set the desired number of ticks
    # cbar.locator = ticker.MaxNLocator(nbins = num_ticks)  # Set the locator
    # cbar.update_ticks()  # Update the ticks on the color bar
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f')) 

    ax3.plot(a_mesh[arg_min_delta_ae], b_mesh[arg_min_delta_ae], 'ro', alpha = 0.2)

    plt.savefig('ae_delta_nfp_{}_r_{}_omn_{}_omt_{}.png'.format(nfp, r, omn, omt), dpi = 300, bbox_inches='tight')

    plt.show()
    

    idx = arg_min_delta_ae

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
        NAE_AE.plot_AE_per_lam()
        
    except:
        print("jaime")

    stel.plot_boundary(r = r)


"""