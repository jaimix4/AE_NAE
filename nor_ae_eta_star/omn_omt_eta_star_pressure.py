import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qsc import Qsc
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_nor_func_pyqsc import ae_nor_nae
from opt_eta_star import set_eta_star
from label_lines_pressure_plot import labelLine, labelLines
from scipy.optimize import minimize, shgo
import time
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
from AEpy import ae_routines as ae

from   matplotlib        import rc
# add latex fonts to plots
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 22})
rc('text', usetex=True)

start_time = time.time()

# A configuration is defined with B0, etabar, field periods, Fourier components and eta bar

B0 = 1        # [T] strength of the magnetic field on axis

nfp = 3       # field periods

eta = -0.1    # eta bar

##########################################################################################

# these quantities are provided in [m]

a = 0.045
b = 0

rc = [1, a, b]
zs = [0, a, b]

##########################################################################################

# set eta star
# stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=-0.1, B0=B0)
# stel = Qsc.from_paper('precise QH')#, nphi = nphi)
# eta_star = set_eta_star(stel)
# print(stel.spsi)
# to calculate things the lam_res needs to be provided, just that?
lam_res = 1002
# r = 0.001
#######


nphi = int(1e3+1)
stel = Qsc.from_paper('precise QH', nphi = nphi)
# stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=-0.1, B0=B0)
stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=-0.1, B0=B0)
eta_star = set_eta_star(stel)
# stel.spsi = -1
# stel.zs = -stel.zs
omnigenous = False

# distance from the magnetic axis to be used

stel.r = r = 0.001
stel.calculate()

# alpha = 0.0
# NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=alpha, N_turns=3, nphi=nphi,
#                      lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')

a_r = 1

# normalization variable for AE

Delta_r = 1

# defining mesh of gradients

N = 30

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
pressure_arr = np.empty_like(omt_mesh)

# read npy file
# try:

#     ae_nor_total_arr = np.load('data_plot/ae_nor_total_arr_omn_omt_{:.4f}_{}_{}_etastar.npy'.format(om_min, om_max, N))
#     eta_dagger_arr = np.load('data_plot/eta_dagger_arr_omn_omt_r_{:.4f}_{}_{}_etastar.npy'.format(om_min, om_max, N))
#     ald_computed = True
#     print("files found")

# except:
#     ald_computed = False
#     print("files not found")


# eta_start = [-0.5]

# simple  minimazation, better for just optimizing eta

counter = 0
total_counter = len(omt_arr)*len(omn_arr)


def ae_nor_total_function_eta(x):

    #return ae_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_psiA, omn, omT, plot = False)[-1]
    return -1*ae_nor_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_r, a_r, omn, omt, plot = False)[-1]

#if ald_computed == False:

for idx, omt in enumerate(omt_arr):

    #eta_start = [-0.5]
    
    for idx_omn, omn in enumerate(omn_arr):


        #result = minimize(ae_nor_total_function_eta, eta_start, method='L-BFGS-B', bounds = [[-500, -0.001]], options={'disp': False})
        #result = shgo(ae_nor_total_function_eta, bounds = [[-400, -0.001]])
        # results with out log output
        #     result = minimize(ae_total_function_eta, eta_start, method='L-BFGS-B', bounds = [[-5, 1]], options={'disp': False})
        #solution = result['x']

        eta = [eta_star]


        eta_dagger_arr[idx_omn, idx] = eta[0]

        ae_nor_total_arr[idx_omn, idx] = -1*ae_nor_total_function_eta(eta)


        # d

        #print('one more done')

        pressure_arr[idx_omn, idx] = omt + omn

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
                


# min_eta_dagger = np.min(-eta_dagger_arr)
# print(min_eta_dagger)

# save the arrays to npy file
# np.save('data_plot/ae_nor_total_arr_omn_omt_{:.4f}_{}_{}_etastar.npy'.format(om_min, om_max, N), ae_nor_total_arr)
# np.save('data_plot/etastar_arr_omn_omt_r_{:.4f}_{}_{}_etastar.npy'.format(om_min, om_max, N), eta_dagger_arr)

# subplots 
#fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#fig, axs = plt.subplots(figsize=(9, 8))
fig, axs = plt.subplots(figsize=(10.8/1.5, 8/1.5))
# AE plot
# countourf plot with z value on log scale color bar as well
# cs = axs[0].contourf(omt_mesh, omn_mesh, np.log10(ae_nor_total_arr), 1000, cmap = mpl.colormaps['plasma'])
cs = axs.contourf(omt_mesh, omn_mesh, np.log10(ae_nor_total_arr), 80, cmap = mpl.colormaps['jet'], alpha = 0.9)
axs.set_xscale('log')
axs.set_yscale('log')
cb = fig.colorbar(cs, ax=axs, shrink=0.9)
cb.set_label(r'$\log_{10}(\hat{A})$',fontsize=25)
# labels
axs.set_ylabel(r'$\hat{\omega}_{n}$', fontsize=28)
axs.set_xlabel(r'$\hat{\omega}_{T}$', fontsize=28)

# axs[0].set_title('a = {}, b = {}, nfp = {}, B0 = {} \n'.format(a, b, nfp, B0) + ' ' + r'$r = $' + '{}'.format(r) + ', ' + \
#     r'$\bar{\eta} = \bar{\eta}_{*} = $' + '{:.4f}'.format(eta_star), fontsize=16)
# axs.set_title(r'$\log_{10}(\hat{A})$' + ', r = {}'.format(r),fontsize=18)
# axs.set_title('a = {}, b = {}, Nfp = {}, B0 = {} '.format(a, b, nfp, B0) + ' \n' + r'$r = $' + '{}'.format(r) + ', ' + \
#     r'$\bar{\eta} = \bar{\eta}_{*} = $' + '{:.4f} \n'.format(eta_star), size=18)

# how to adjust the font size of the ticks

axs.tick_params(axis='both', which='major', labelsize=22)

# how to adjust the font size of the tick labels in a colorbar
cb.ax.tick_params(labelsize=22)

# set limits of plot 
axs.set_xlim(omt_arr[0], omt_arr[-1])
axs.set_ylim(omn_arr[0], omn_arr[-1])

# PRESSURE LINES
# omt array vs :
angles = np.linspace(0, np.pi/2, 100)
omp_arr = np.array([0.5, 1, 5, 10, 25, 50])
two_thirds = np.arctan(np.sqrt(3/2))
for omp in omp_arr:
    omt_pre = omp*(np.cos(angles))**2
    omn_pre = omp*(np.sin(angles))**2
    axs.plot(omt_pre, omn_pre, color='black', linestyle='-.', linewidth=1, alpha = 0.6, label = r'$\hat{\omega}_{p}$' + ' = {}'.format(omp))
    #axs.scatter(omp*(np.cos(two_thirds))**2, omp*(np.sin(two_thirds))**2, alpha=1, label = 'omp = {}'.format(omp))

axs.plot((2/3)*omt_arr, omn_arr, color='black', linestyle='--', linewidth=1, alpha = 0.5, label = r'$\eta = 2/3$')

# xvals = [0.5, 1, 5, 10, 25, 50]
# xvals = np.concatenate((np.array([1.5]), omp_arr * 0.5))
# tailor for graph tbh
xvals = np.array([0.5, 1, 5, 10, 25, 50, 1.5])/1.5
labelLines(plt.gca().get_lines(), xvals = xvals)


# 2/3 line eta = omt/omn < 2/3

# axs.text((2/3)*0.5 + 0.2, 0.5, r'$\frac{omt}{omn} = \frac{2}{3}$', fontsize=18, color='black', transform=axs.transAxes)
# axs.legend(loc = 'upper right', fontsize=8)

"""
# eta dagger plot
cs = axs[1].contourf(omt_mesh, omn_mesh, np.log10(pressure_arr), 80, cmap = mpl.colormaps['jet'])
axs[1].set_xscale('log')
axs[1].set_yscale('log')
fig.colorbar(cs, ax=axs[1], shrink=0.9)
# labels
axs[1].set_title(r'$\log_{10}($' + 'omp' + r'$)$', fontsize=20)
# title
axs[1].set_xlabel('omt', fontsize=18)

# 2/3 line eta = omt/omn < 2/3

axs[1].plot((2/3)*omt_arr, omn_arr, color='black', linestyle='--', linewidth=1, alpha = 0.5)
axs[1].text((2/3)*0.5 + 0.2, 0.5, r'$\frac{omt}{omn} = \frac{2}{3}$', fontsize=18, color='black', transform=axs[1].transAxes)

axs[1].set_xlim(omt_arr[0], omt_arr[-1])
axs[1].set_ylim(omn_arr[0], omn_arr[-1])
"""


# title for the whole figure
# plt.suptitle('a = {}, b = {}, nfp = {}, B0 = {} '.format(a, b, nfp, B0) + ' ' + r'$r = $' + '{}'.format(r) + ', ' + \
#     r'$\bar{\eta} = \bar{\eta}_{*} = $' + '{:.4f}'.format(eta_star), size=16)



fig.savefig('THESIS_etastar_omn_omt_r_{}_N_{}_eta_{:.2f}pressure_pres_second.png'.format(r, N, np.abs(eta_star)), format='png', dpi=300, bbox_inches='tight')
plt.show()

