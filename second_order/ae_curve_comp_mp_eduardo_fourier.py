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

rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

# set configuration space
B0 = 1
nfp = 3


N = 150
# NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
a_arr = np.linspace(0.0001, 0.3, N)
b_arr = np.linspace(-0.06, 0.06, N)

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

# getting straight arrays of a and b

delB20_str_arr = np.zeros_like(a_arr)
r_crit_str_arr = np.zeros_like(a_arr)
ae_first_arr = np.zeros_like(a_arr)
ae_second_arr = np.zeros_like(a_arr)

# ae params

nphi = int(1e3+1)
lam_res = 2001
omn = 3
omt = 0
omnigenous = False

r = 1e-3

try:

    ae_first_arr = np.load('data_curve/ae_first_curve_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}.npy'.format(N, nfp, r, omn, omt))
    ae_second_arr = np.load('data_curve/ae_second_curve_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}.npy'.format(N, nfp, r, omn, omt))

    computed = True
    print("files found")
except:

    computed = False
    print("files not found")



if computed == False:

    for i in range(len(idx_curve)):

        stel = Qsc(rc=[1, a_arr[i], b_arr[idx_curve[i][1]]], zs=[0.0, a_z_arr[idx_curve[i][::-1]], b_z_arr[idx_curve[i][::-1]]], \
            nfp=nfp, etabar=eta_arr[idx_curve[i][::-1]], B0=B0, order = "r2", B2c = B2c_arr[idx_curve[i][::-1]], nphi=nphi)
        stel.spsi = 1
        stel.calculate()
        alpha = 1.0
        stel.r = r 
        try:
            NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=alpha, N_turns=3, nphi=nphi,
                                    lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
            NAE_AE.calc_AE(omn=stel.spsi*omn,omt=stel.spsi*omt,omnigenous=omnigenous)
            #NAE_AE.plot_AE_per_lam()
            ae_second_arr[i] = NAE_AE.ae_tot
        except:
            print('could not compute AE')
            ae_second_arr[i] = np.nan

        ae_first_arr[i] = ae_nor_nae(stel, r, lam_res, 1, 1, omn, omt, plot = False)[-1]
    

for i in range(len(idx_curve)):

    delB20_str_arr[i] = delB20_arr[idx_curve[i][::-1]]
    r_crit_str_arr[i] = rcrit_arr[idx_curve[i][::-1]]



##########################################################################################
# save curve arrays

# ae_first_arr = np.load('data_curve/ae_first_curve_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}.npy'.format(N, nfp, r, omn, omt))
# ae_second_arr = np.load('data_curve/ae_second_curve_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}.npy'.format(N, nfp, r, omn, omt))

np.save('data_curve/ae_first_curve_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}.npy'.format(N, nfp, r, omn, omt), ae_first_arr)
np.save('data_curve/ae_second_curve_arr_N_{}_nfp_{}_r_{}_omn_{}_omt_{}.npy'.format(N, nfp, r, omn, omt), ae_second_arr)

##########################################################################################

fig = plt.figure(figsize = (7, 6))

plt.plot(a_arr, delB20_str_arr)
# plt.ylim(0, 0.7)

for i in range(len(idx_curve)):
    if i == 0:
        plt.plot(a_arr[idx_curve[i][0]], delB20_arr[idx_curve[i][::-1]], 'k.', markersize = 3, label = 'curve points')
    # elif i == 59:
    #     plt.plot(a_arr[idx_curve[i][0]], delB20_arr[idx_curve[i][::-1]], 'r.', markersize = 6)
        # plt.legend()
    plt.plot(a_arr[idx_curve[i][0]], delB20_arr[idx_curve[i][::-1]], 'k.', markersize = 3)

plt.yscale('log')
plt.xlabel('a')
plt.ylabel(r'$\Delta B_{20}$')




fig = plt.figure(figsize = (7, 6))

plt.plot(a_arr, ae_first_arr, 'r*-', label = 'AE first')
plt.plot(a_arr, ae_second_arr, 'b-', label = 'AE second')
plt.title('AE at r = {}, omn = {}, omt = {}'.format(r, omn, omt))
plt.legend()


##########################################################################################

# give the index of the values from delB20_arr with filtering of values less than 0
idx = np.where(delB20_arr < 0.7)
# now make a tuple numpy array of the indices
idx = list(zip(idx[0], idx[1]))


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


##########################################################################################

# idx_min = np.unravel_index(np.argmin(delB20_arr, axis=None), delB20_arr.shape)
# print(idx_min)
# print(np.min(delB20_arr))
# print(a_mesh[idx_min[0], idx_min[1]])
# print(b_mesh[idx_min[0], idx_min[1]])
# # print(np.argmin(delB20_arr))

plt.show()

# # fig = plt.figure(figsize = (7, 6))

# # levels = np.linspace(np.min(eta_arr), np.max(eta_arr), 300)
# # cs = plt.contourf(a_mesh, b_mesh, eta_arr, levels = levels, cmap = mpl.colormaps['jet'])
# # fig.colorbar(cs, shrink=0.9)
# # plt.title('nfp = {},  B0 = {},  N = {},  '.format(nfp, B0, N) + r'$\bar{\eta}$')



# # # plt.show()

# # give the index of the values from delB20_arr with filtering of values less than 0
# idx = np.where(rcrit_arr > 0.05)
# # now make a tuple numpy array of the indices
# idx = list(zip(idx[0], idx[1]))

# # print(idx)

# # fig = plt.figure(figsize = (7, 6))


# # # levels = np.linspace(0.01, 0.2, 300)
# # levels = np.linspace(1e-4, 0.2, 300)
# # # levels = np.linspace(0.01, 1, 300)
# # cs = plt.contourf(a_mesh, b_mesh, rcrit_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'max')
# # fig.colorbar(cs, shrink=0.9)
# # plt.title('nfp = {},  B0 = {},  N = {},          '.format(nfp, B0, N) + r'$ r_{\rm crit.}$')

# # # plot points where rcrit > 0.1
# # for i in range(len(idx)):
# #     if i == 0:
# #         plt.plot(a_mesh[idx[i][0], idx[i][1]], b_mesh[idx[i][0], idx[i][1]], 'k.', markersize = 2, alpha = 0.3, label = r'$r_{\rm crit.} > 0.05$')
# #     plt.plot(a_mesh[idx[i][0], idx[i][1]], b_mesh[idx[i][0], idx[i][1]], 'k.', markersize = 2, alpha = 0.3)

# # plt.plot(a_arr, curve(a_arr), 'k--', label = 'fit')

# # plt.xlabel('a')
# # plt.ylabel('b')
# # plt.legend(loc = 'upper left', fontsize = 12)
# # display plot in current screen
# plt.show()

