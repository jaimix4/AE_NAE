import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qsc import Qsc
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_nor_func_pyqsc import ae_nor_nae
from scipy.optimize import minimize, shgo
import time
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm

start_time = time.time()

# A configuration is defined with B0, etabar, field periods, Fourier components and eta bar

B0 = 1        # [T] strength of the magnetic field on axis

nfp = 3       # field periods

eta = -0.1    # eta bar

##########################################################################################

# these quantities are provided in [m]

rc = [1, 0.045]
zs = [0, 0.045]

# to calculate things the lam_res needs to be provided, just that?
lam_res = 10000

# distance from the magnetic axis to be used

r = 0.001

a_r = 1

# normalization variable for AE

Delta_r = 1

# defining mesh of gradients

N = 60

om_max = 100
om_min = 1e-1
# omt_arr = np.linspace(om_min, om_max, N)
# omn_arr = np.linspace(om_min, om_max, N)
omt_arr = np.geomspace(om_min, om_max, N)
omn_arr = np.geomspace(om_min, om_max, N)

# mesh grid omt and omn arrs
omt_mesh, omn_mesh = np.meshgrid(omt_arr, omn_arr)

# empty array with shape N by N_eta
ae_nor_total_arr = np.empty_like(omt_mesh)
eta_dagger_arr = np.empty_like(omt_mesh)

# read npy file
try:

    ae_nor_total_arr = np.load('data_plot/shgo_ae_nor_total_arr_omn_omt_{:.4f}_{}_{}.npy'.format(om_min, om_max, N))
    eta_dagger_arr = np.load('data_plot/shgo_eta_dagger_arr_omn_omt_r_{:.4f}_{}_{}.npy'.format(om_min, om_max, N))
    ald_computed = True
    print("files found")

except:
    ald_computed = False
    print("files not found")


eta_start = [-1]

# simple  minimazation, better for just optimizing eta

counter = 0
total_counter = len(omt_arr)*len(omn_arr)


def ae_nor_total_function_eta(x):

    #return ae_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_psiA, omn, omT, plot = False)[-1]
    return -1*ae_nor_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_r, a_r, omn, omt, plot = False)[-1]

if ald_computed == False:


    for idx, omt in enumerate(omt_arr):
        
        for idx_omn, omn in enumerate(omn_arr):


            #result = minimize(ae_nor_total_function_eta, eta_start, method='L-BFGS-B', bounds = [[-500, -0.001]], options={'disp': False})
            result = shgo(ae_nor_total_function_eta, bounds = [[-300, -0.001]])
            # results with out log output
            #     result = minimize(ae_total_function_eta, eta_start, method='L-BFGS-B', bounds = [[-5, 1]], options={'disp': False})
            solution = result['x']

            eta_dagger_arr[idx_omn, idx] = solution[0]

            #eta_start = [solution[0]]

            ae_nor_total_arr[idx_omn, idx] = -1*ae_nor_total_function_eta(solution)

            counter += 1
            
            if counter % 100 == 0:

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
                


min_eta_dagger = np.min(-eta_dagger_arr)
print(min_eta_dagger)

# save the arrays to npy file
np.save('data_plot/shgo_ae_nor_total_arr_omn_omt_{:.4f}_{}_{}.npy'.format(om_min, om_max, N), ae_nor_total_arr)
np.save('data_plot/shgo_eta_dagger_arr_omn_omt_r_{:.4f}_{}_{}.npy'.format(om_min, om_max, N), eta_dagger_arr)

# subplots 
fig, axs = plt.subplots(1, 2, figsize=(13, 6))

# AE plot
# countourf plot with z value on log scale color bar as well
# cs = axs[0].contourf(omt_mesh, omn_mesh, np.log10(ae_nor_total_arr), 1000, cmap = mpl.colormaps['plasma'])
cs = axs[0].contourf(omt_mesh, omn_mesh, np.log10(ae_nor_total_arr), 50, cmap = mpl.colormaps['jet'])
axs[0].set_xscale('log')
axs[0].set_yscale('log')
fig.colorbar(cs, ax=axs[0], shrink=0.9)
# labels
axs[0].set_ylabel('omn', fontsize=18)
axs[0].set_xlabel('omt', fontsize=18)
axs[0].set_title(r'$\log_{10}(\hat{A})$', fontsize=20)

# eta dagger plot
cs = axs[1].contourf(omt_mesh, omn_mesh, np.log10(-1*eta_dagger_arr), 50, cmap = mpl.colormaps['jet'])
axs[1].set_xscale('log')
axs[1].set_yscale('log')
fig.colorbar(cs, ax=axs[1], shrink=0.9)
# labels
axs[1].set_title(r'$\log_{10}(-\bar{\eta}_{\dag})$', fontsize=20)
axs[1].set_xlabel('omt', fontsize=18)

# title for the whole figure
plt.suptitle('r = {}\n'.format(r), size = 20)

fig.savefig('shgo_eta_dagger_omn_omt_r_{}_{}_geo.png'.format(r, om_max), format='png', dpi=300, bbox_inches='tight')
plt.show()

