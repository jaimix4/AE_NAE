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


# set configuration space
B0 = 1
nfp = 3


N = 20
# NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
a_arr = np.linspace(0.0001, 0.3, N)
b_arr = np.linspace(-0.06, 0.06, N)

# mesh grid a_arr and b_arr
a_mesh, b_mesh = np.meshgrid(a_arr, b_arr)

shape_mesh = a_mesh.shape

a_z_arr = np.full(shape_mesh, np.nan)
b_z_arr = np.full(shape_mesh, np.nan)
eta_arr = np.full(shape_mesh, np.nan)
B2c_arr = np.full(shape_mesh, np.nan)

delB20_arr = np.full(shape_mesh, np.nan)


try:

    a_z_arr = np.load('shapes_2nd/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    b_z_arr = np.load('shapes_2nd/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    eta_arr = np.load('shapes_2nd/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    B2c_arr = np.load('shapes_2nd/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    delB20_arr = np.load('shapes_2nd/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp))

    ald_computed = True
    print("files found")

except:

    ald_computed = False
    print("files not found")

if ald_computed == False:

    counter = 0
    total_counter = len(a_arr)*len(b_arr)
    start_time = time.time()

    for idx, a in enumerate(a_arr):

            
            for idx_b, b in enumerate(b_arr):


                rcOr=[1, a, b]
                zsOr=[0.0, a, b]

                stel = Qsc(rc=rcOr, zs=zsOr, B0 = B0, nfp=nfp, order='r1', nphi = 61)

                num_iters = choose_Z_axis(stel, max_num_iter=50)

                print(num_iters)

                a_z_arr[idx_b, idx] = stel.zs[1]
                b_z_arr[idx_b, idx] = stel.zs[2]
                eta_arr[idx_b, idx] = stel.etabar
                B2c_arr[idx_b, idx] = stel.B2c

                delB20_arr[idx_b, idx] = stel.B20_variation

                counter += 1
                
                if counter % 5 == 0:

                    elapsed_time = time.time() - start_time

                    elapsed_time_min = int(elapsed_time/60) 
                    elapsed_time_sec = int(elapsed_time - elapsed_time_min*60)

                    if elapsed_time_sec < 10:
                        elapsed_time_sec = "0" + str(elapsed_time_sec)

                    eta_time = (elapsed_time/60)*(total_counter/counter-1)

                    eta_time_min = int(eta_time)
                    eta_time_sec = int((eta_time - eta_time_min)*60)

                    if eta_time_sec < 10:
                        eta_time_sec = "0" + str(eta_time_sec)

                    print("###############################################################")
                    print("\nConfigurations done: {}/{}. Elapsed time: {}:{}. ETA: {}:{}. ".format(counter, total_counter, elapsed_time_min, elapsed_time_sec,  eta_time_min, eta_time_sec))
                    # print progress bar
                    print("Progress: [", end="")
                    for i in range(0, int(counter/total_counter*50)):
                        print("=", end="")
                    print(">", end="")
                    for i in range(0, 50-int(counter/total_counter*50)):
                        print(" ", end="")
                    print("] \n")
                    print("###############################################################")


np.save('shapes_2nd/a_arr_N_{}_nfp_{}.npy'.format(N, nfp), a_mesh)
np.save('shapes_2nd/b_arr_N_{}_nfp_{}.npy'.format(N, nfp), b_mesh)
np.save('shapes_2nd/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp), a_z_arr)
np.save('shapes_2nd/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp), b_z_arr)
np.save('shapes_2nd/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp), eta_arr)
np.save('shapes_2nd/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp), B2c_arr)
np.save('shapes_2nd/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp), delB20_arr)


fig = plt.figure(figsize = (7, 6))

levels = np.linspace(np.log10(np.min(delB20_arr)), np.log10(np.max(delB20_arr)), 100)
cs = plt.contourf(a_mesh, b_mesh, np.log10(delB20_arr), levels = levels, cmap = mpl.colormaps['jet'])
fig.colorbar(cs, shrink=0.9)
plt.title('nfp = {},  B0 = {},  N = {},  '.format(nfp, B0, N) + r'$\log_{10} \: \Delta B_{20}$')

plt.xlabel('a')
plt.ylabel('b')

print(delB20_arr)

idx_min = np.unravel_index(np.argmin(delB20_arr, axis=None), delB20_arr.shape)
print(idx_min)
print(np.min(delB20_arr))
print(a_mesh[idx_min[0], idx_min[1]])
print(b_mesh[idx_min[0], idx_min[1]])
# print(np.argmin(delB20_arr))

plt.show()

fig = plt.figure(figsize = (7, 6))

levels = np.linspace(np.min(eta_arr), np.max(eta_arr), 100)
cs = plt.contourf(a_mesh, b_mesh, eta_arr, levels = levels, cmap = mpl.colormaps['jet'])
fig.colorbar(cs, shrink=0.9)
plt.title('nfp = {},  B0 = {},  N = {},  '.format(nfp, B0, N) + r'$\bar{\eta}$')



plt.show()







