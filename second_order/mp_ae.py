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
# r = 0.02

# ok I have to do r = 0.02

####################

N = 150
# nfp = 3
B0 = 1

nfp = 3

# NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
a_arr = np.linspace(0.0001, 0.3, N)
b_arr = np.linspace(-0.06, 0.06, N)

# nfp = 2

# # NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 2
# a_arr = np.linspace(0.0001, 0.6, N)
# b_arr = np.linspace(-0.13, 0.13, N)

# nfp = 4

# # NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 4
# a_arr = np.linspace(0.0001, 0.20, N)
# b_arr = np.linspace(-0.0347, 0.0347, N)



# # NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
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



    levels = np.linspace(np.nanmin(ae_first_arr), np.nanmax(ae_first_arr), 200)
    levels = np.linspace(np.nanmin(ae_second_arr), np.nanmax(ae_second_arr), 200)

    fig = plt.figure()
    plt.contourf(a_mesh, b_mesh, ae_first_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
    plt.colorbar()
    plt.title('AE first at r = {}, omn = {}, omt = {}'.format(r, omn, omt))
    plt.xlabel('a')
    plt.ylabel('b')

    plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)
    plt.savefig('first_ae_{}.png'.format(r), dpi = 300)

    fig = plt.figure()
    # levels = np.linspace(np.nanmin(ae_second_arr), np.nanmax(ae_second_arr), 200)
    plt.contourf(a_mesh, b_mesh, ae_second_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
    plt.title('AE second at r = {}, omn = {}, omt = {}'.format(r, omn, omt))
    plt.colorbar()
    plt.xlabel('a')
    plt.ylabel('b')

    plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)
    plt.savefig('first_ae_{}.png'.format(r), dpi = 300)

    delta_ae = (ae_second_arr/ae_first_arr - 1)#/r

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

    delta_ae[delta_ae > 0] = np.nan
    delta_ae[delta_ae < -0.3] = np.nan
    delta_ae[a_mesh < 0.15] = np.nan


       # index of minimum value on delta_ae
    arg_min_delta_ae = np.nanargmin(delta_ae)

    arg_min_delta_ae = np.unravel_index(np.nanargmin(delta_ae), delta_ae.shape)

    print(arg_min_delta_ae)

    print(delta_ae[arg_min_delta_ae])


    fig = plt.figure()
    levels = np.linspace(-20.0, 20.0, 300)
    levels = np.linspace(-0.5, 0.5, 300)
    # levels = 200
    plt.contourf(a_mesh, b_mesh, delta_ae, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
    plt.title('(AE2nd / AE1st - 1) ,  at r = {}, omn = {}, omt = {}'.format(r, omn, omt))
    plt.colorbar()
    plt.xlabel('a')
    plt.ylabel('b')


    plt.plot(a_arr, b_fit_curve_arr, alpha = 0.2)

    plt.plot(a_mesh[arg_min_delta_ae], b_mesh[arg_min_delta_ae], 'ro', alpha = 0.2)

    plt.savefig('ae_second_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}_mp.png'.format(N, nfp, r, omn, omt), dpi = 300)
    plt.show()
    

    idx = arg_min_delta_ae

    stel = Qsc(rc=[1, a_mesh[idx], b_mesh[idx]], zs=[0.0, a_z_arr[idx], b_z_arr[idx]], \
        nfp=nfp, etabar=eta_arr[idx], B0=B0, order = "r2", B2c = B2c_arr[idx], nphi=nphi)
    stel.spsi = 1
    stel.calculate()

    stel.plot_boundary(r = r)


