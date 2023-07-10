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



def opt_stel_func(a,b):
    stel = Qsc(rc=[1, a, b], zs=[0, a, b], B0 = 1.0, nfp=nfp, order='r1', nphi = 61)
    num_iters = choose_Z_axis(stel, max_num_iter=50)
    return stel



N = 150


# nfp = 3

# # NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
# a_arr = np.linspace(0.0001, 0.3, N)
# b_arr = np.linspace(-0.06, 0.06, N)

# nfp = 2

# # NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
# a_arr = np.linspace(0.0001, 0.6, N)
# b_arr = np.linspace(-0.13, 0.13, N)

nfp = 4

# NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
a_arr = np.linspace(0.0001, 0.20, N)
b_arr = np.linspace(-0.0347, 0.0347, N)

# mesh grid a_arr and b_arr
a_mesh, b_mesh = np.meshgrid(a_arr, b_arr)

shape_mesh = a_mesh.shape
a_z_arr = np.full(shape_mesh, np.nan)
b_z_arr = np.full(shape_mesh, np.nan)
eta_arr = np.full(shape_mesh, np.nan)
B2c_arr = np.full(shape_mesh, np.nan)
delB20_arr = np.full(shape_mesh, np.nan)
rcrit_arr   = np.empty_like(a_mesh)


if __name__ == "__main__":
    num_cores = 9
    pool = mp.Pool(num_cores)
    print('Number of cores used: {}'.format(num_cores))


    # now loop over a b
    print('optimising stellarators')
    start_time = time.time()
    output_list = pool.starmap(opt_stel_func, [(a_mesh[idx],b_mesh[idx]) for idx, _ in np.ndenumerate(a_mesh)])
    print("data generated in       --- %s seconds ---" % (time.time() - start_time))
    # close pool
    pool.close()
    # transfer data to arrays
    list_idx = 0

    # print(output_list)

    for idx, _ in np.ndenumerate(a_mesh):
        stel = output_list[list_idx]
        #eta_arr[idx] = stel.etabar
        a_z_arr[idx] = stel.zs[1]
        b_z_arr[idx] = stel.zs[2]
        eta_arr[idx] = stel.etabar
        B2c_arr[idx] = stel.B2c
        delB20_arr[idx] = stel.B20_variation
        rcrit_arr[idx]   = stel.r_singularity

        list_idx = list_idx + 1

    np.save('shapes_2nd/a_arr_N_{}_nfp_{}.npy'.format(N, nfp), a_mesh)
    np.save('shapes_2nd/b_arr_N_{}_nfp_{}.npy'.format(N, nfp), b_mesh)
    np.save('shapes_2nd/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp), a_z_arr)
    np.save('shapes_2nd/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp), b_z_arr)
    np.save('shapes_2nd/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp), eta_arr)
    np.save('shapes_2nd/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp), B2c_arr)
    np.save('shapes_2nd/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp), delB20_arr)
    np.save('shapes_2nd/rcrit_arr_N_{}_nfp_{}.npy'.format(N, nfp), rcrit_arr)

    plt.contourf(a_mesh, b_mesh, eta_arr)
    plt.show()




