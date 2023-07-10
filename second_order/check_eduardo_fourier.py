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


N = 150
# NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
a_arr = np.linspace(0.0001, 0.3, N)
b_arr = np.linspace(-0.06, 0.06, N)

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



try:
    a_mesh = np.load('shapes_2nd/a_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    b_mesh = np.load('shapes_2nd/b_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    a_z_arr = np.load('shapes_2nd/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    b_z_arr = np.load('shapes_2nd/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    eta_arr = np.load('shapes_2nd/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    B2c_arr = np.load('shapes_2nd/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    delB20_arr = np.load('shapes_2nd/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    rcrit_arr = np.load('shapes_2nd/rcrit_arr_N_{}_nfp_{}.npy'.format(N, nfp))

    ald_computed = True
    print("files found")

except:

    ald_computed = False
    print("files not found")


##########################################################################################
# finding a curve to try some things

a_fit_arr = [0.0001, 0.0227, 0.0505, 0.0946, 0.1817, 0.2012, 0.2740, 0.2983]
b_fit_arr = [0.01163, 2e-5, 0.00091, 0.00592, 0.01058, 0.01372, 0.00662, -0.04910]

curve = np.polyfit(a_fit_arr, b_fit_arr, 7)

# use the curve to write a lambda function that plots this curve
curve = np.poly1d(curve)

# print(curve(a_arr))

# get the b_arr closest to the curve
b_fit_curve_arr = curve(a_arr)

# get the all the index of b_mesh that are the closes to b_fit_curve_arr
idx_curve = []
for i in range(len(b_fit_curve_arr)):
    idx_curve.append((i, np.argmin(np.abs(b_arr - b_fit_curve_arr[i]))))
# idx = np.array(idx_cruve)

print(idx_curve)


##########################################################################################

# give the index of the values from delB20_arr with filtering of values less than 0
idx = np.where(delB20_arr < 0.7)
# now make a tuple numpy array of the indices
idx = list(zip(idx[0], idx[1]))

# print(idx)


idx_min = np.unravel_index(np.argmin(delB20_arr, axis=None), delB20_arr.shape)
# print(idx_min)

fig = plt.figure(figsize = (7, 6))

# levels = np.linspace(np.log10(np.min(delB20_arr)), np.log10(np.max(delB20_arr)), 300)
levels = np.linspace(np.log10(np.min(delB20_arr)), 0, 300)
# levels = np.linspace(np.log10(np.min(delB20_arr)), 6, 200)
cs = plt.contourf(a_mesh, b_mesh, np.log10(delB20_arr), levels = levels, cmap = mpl.colormaps['jet'])
fig.colorbar(cs, shrink=0.9)
plt.title('nfp = {},  B0 = {},  N = {},         '.format(nfp, B0, N) + r'$\log_{10} \: \Delta B_{20}$')

plt.xlabel('a')
plt.ylabel('b')

# # plot points where delB20 < 0.1
for i in range(len(idx)):
    if i == 0:
        plt.plot(a_mesh[idx[i][0], idx[i][1]], b_mesh[idx[i][0], idx[i][1]], 'k.', markersize = 2, alpha = 0.3, label = r'$\Delta B_{20} < 0.7$')
    plt.plot(a_mesh[idx[i][0], idx[i][1]], b_mesh[idx[i][0], idx[i][1]], 'k.', markersize = 2, alpha = 0.3)



plt.plot(a_arr, curve(a_arr), 'k--', label = 'cruve fit')
plt.plot(a_fit_arr, b_fit_arr, 'k.', label = 'fit points')

for i in range(len(idx_curve)):
    if i == 0:
        plt.plot(a_arr[idx_curve[i][0]], b_arr[idx_curve[i][1]], 'k.', markersize = 3, label = 'curve points')
    plt.plot(a_arr[idx_curve[i][0]], b_arr[idx_curve[i][1]], 'k.', markersize = 3)

plt.legend(loc = 'upper right', fontsize = 12)

# idx_min = np.unravel_index(np.argmin(delB20_arr, axis=None), delB20_arr.shape)
# print(idx_min)
# print(np.min(delB20_arr))
# print(a_mesh[idx_min[0], idx_min[1]])
# print(b_mesh[idx_min[0], idx_min[1]])
# # print(np.argmin(delB20_arr))

# plt.show()

# fig = plt.figure(figsize = (7, 6))

# levels = np.linspace(np.min(eta_arr), np.max(eta_arr), 300)
# cs = plt.contourf(a_mesh, b_mesh, eta_arr, levels = levels, cmap = mpl.colormaps['jet'])
# fig.colorbar(cs, shrink=0.9)
# plt.title('nfp = {},  B0 = {},  N = {},  '.format(nfp, B0, N) + r'$\bar{\eta}$')



# # plt.show()

##########################################################################################
# plot of r_crit

# give the index of the values from delB20_arr with filtering of values less than 0
idx = np.where(rcrit_arr > 0.05)
# now make a tuple numpy array of the indices
idx = list(zip(idx[0], idx[1]))

# print(idx)

fig = plt.figure(figsize = (7, 6))


# levels = np.linspace(0.01, 0.2, 300)
levels = np.linspace(1e-4, 0.2, 300)
# levels = np.linspace(0.01, 1, 300)
cs = plt.contourf(a_mesh, b_mesh, rcrit_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'max')
fig.colorbar(cs, shrink=0.9)
plt.title('nfp = {},  B0 = {},  N = {},          '.format(nfp, B0, N) + r'$ r_{\rm crit.}$')

# plot points where rcrit > 0.1
for i in range(len(idx)):
    if i == 0:
        plt.plot(a_mesh[idx[i][0], idx[i][1]], b_mesh[idx[i][0], idx[i][1]], 'k.', markersize = 2, alpha = 0.3, label = r'$r_{\rm crit.} > 0.05$')
    plt.plot(a_mesh[idx[i][0], idx[i][1]], b_mesh[idx[i][0], idx[i][1]], 'k.', markersize = 2, alpha = 0.3)

plt.plot(a_arr, curve(a_arr), 'k--', label = 'fit')

plt.xlabel('a')
plt.ylabel('b')
plt.legend(loc = 'upper left', fontsize = 12)

##########################################################################################


# display plot in current screen
plt.show()

