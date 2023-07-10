import sys
import numpy as np
import matplotlib.pyplot as plt
from qsc import Qsc
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_nor_func_pyqsc import ae_nor_nae
from opt_eta_star import set_eta_star
from ae_func_pyqsc import ae_nae
from scipy.optimize import minimize
import time

# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True

# from Rodriguez paper
# a goes from -0.3 to 0.3
a = 0.045

# b goes from -0.06 to 0.06
b = -0.05638190954773869

rc = np.array([1, a, b])

zs = np.array([0, a, b])

nfp = 3

eta = -0.9

B0 = 1

lam_res = 100000

Delta_r = 1
a_r = 1

# omt_omn_arr = np.linspace(0.01, 1/3, 2/310)
# omt_omn_arr = np.array([1/5, 2/3, 10])
# omt_omn_arr_str = ["1/5", "2/3", "10"]
omt_omn_arr = np.array([2/3, 10])
omt_omn_arr_str = ["2/3", "10"]
omt_omn_len = len(omt_omn_arr)

# colors_plot_ae = ['-*r', '-*c', '-*y']
# colors_plot_ae_nor = ['-.r', '--c', '--y']

colors_plot_ae = ['-*c', '-*y']
colors_plot_ae_nor = ['--c', '--y']

r = 1e-5

N_a = 500

a_arr = np.linspace(0.01, 0.3, N_a)
# empty array with shape omt_omn_len x N_a
ae_total_arr = np.empty_like(a_arr)
ae_nor_total_arr = np.empty_like(a_arr)
# ae_total_arr = np.empty((omt_omn_len, N_a))
# ae_total_arr = np.empty((omt_omn_len, N_a))
eta_arr = np.empty_like(a_arr)
alpha_tilde_arr = np.empty_like(a_arr)

iota_arr = np.empty_like(a_arr)

eta_start = [-0.7]

# simple  minimazation, better for just optimizing eta

def ae_total_function_eta(x):

    #return ae_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_psiA, omn, omT, plot = False)[-1]
    return ae_nor_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_r, a_r, omn, omt, plot = False)[-2]


def ae_nor_total_function_eta(x):

    #return ae_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_psiA, omn, omT, plot = False)[-1]
    #print(omn)
    return ae_nor_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_r, a_r, omn, omt, plot = False)[-1]

# define figure

fig,ax = plt.subplots(figsize=(11,6))




counter = 0
total_counter = N_a*len(omt_omn_arr)

start_time = time.time()

for idx_om, omt_omn in enumerate(omt_omn_arr):

    pressure = 5

    omn = pressure/(omt_omn + 1)

    omt = pressure - omn 

    for idx, a in enumerate(a_arr):

        rc = np.array([1, a, b])

        zs = np.array([0, a, b])

        eta = -0.9

        stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)

        eta_star = set_eta_star(stel)

        eta_arr[idx] = eta_star

        ae_total_arr[idx] = ae_total_function_eta([eta_star])
        ae_nor_total_arr[idx] = ae_nor_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta_star, B0=B0), r, lam_res, Delta_r, a_r, omn, omt, plot = False)[-1]

        stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta_arr[idx], B0=B0)

        alpha_tilde_arr[idx] = stel.iotaN - stel.iota

        #iota_arr[idx] = np.abs(stel.G0/stel.iotaN)
        iota_arr[idx] = np.abs(stel.iotaN)


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
            print("\nConfigurations done: {}/{}. Elapsed time: {}:{}. ETA: {}:{}. ".format(counter, total_counter, elapsed_time_min, elapsed_time_sec, eta_time_min, eta_time_sec))
            # print progress bar
            print("Progress: [", end="")
            for i in range(0, int(counter/total_counter*50)):
                print("=", end="")
            print(">", end="")
            for i in range(0, 50-int(counter/total_counter*50)):
                print(" ", end="")
            print("] \n")
            print("###############################################################", end='\r')

    
    # ax.plot(a_arr, ae_total_arr, '-*', zorder = 3, linewidth = 2, markersize = 0.2, label = r'$\hat{A}$' + ' {:.4f}'.format(omt_omn))
    # ax.plot(a_arr, ae_nor_total_arr, '--', zorder = 3, linewidth = 2, markersize = 0.2, label = r'$\hat{A}/E_t$')
    if colors_plot_ae[idx_om] == '-*r':
        ax.plot(a_arr, ae_total_arr, colors_plot_ae[idx_om], zorder = 3, linewidth = 1.5, markersize = 4.0, label = r'$\hat{A}$' + ' omn = {:.2f}, '.format(omn) + r'$\eta = $' + '{}'.format(omt_omn_arr_str[idx_om]))
    else:
        ax.plot(a_arr, ae_total_arr, colors_plot_ae[idx_om], zorder = 3, linewidth = 1.5, markersize = 0.8, label = r'$\hat{A}$' + ' omn = {:.2f}, '.format(omn) + r'$\eta = $' + '{}'.format(omt_omn_arr_str[idx_om]))
    ax.plot(a_arr, ae_nor_total_arr, colors_plot_ae_nor[idx_om], zorder = 3, linewidth = 1.5, markersize = 0.8, label = r'$\hat{A}/E_t$' + ' omt = {:.2f}'.format(omt))

## function for labeling cool lines 

# labelLines(plt.gca().get_lines(),zorder=2.5)

ax.legend(loc = 'upper right', fontsize = 12).set_zorder(4)

ax.set_yscale('log')
ax.grid(alpha = 0.4)

# ax.set_ylabel(r'$\hat{A}$', size = 22, color = 'r')
ax.set_xlabel('a (for axis shape)', size = 20)
ax.set_xlim(np.min(a_arr), np.max(a_arr))
ax.set_title('Scan over axis shape: Rc = [1, a], Zs = [0, a] \n' + r'$\bar{\eta} = \bar{\eta}_{*}$' + \
    '(max $\iota_N$), ' + ' B0 = {}, r = {}, omp = {},     - nfp = {} -'.format(B0, r, pressure, nfp), size = 18)

# plotting second axis

ax2=ax.twinx()

ax3 = ax.twinx()

ax2.plot(a_arr, eta_arr, '-*b', zorder = 3, linewidth = 2, markersize = 0.2)

ax3.plot(a_arr, iota_arr, '-*g', zorder = 3, linewidth = 2, markersize = 0.2)

ax3.spines['right'].set_position(('axes', 1.13))

ax3.set_yscale('log')

ax2.set_ylabel(r'$\bar{\eta}_{*}$', size = 22, color = 'b')

#ax3.set_ylabel(r'$|G0/\iota_N|$', size = 22, color = 'g')

ax3.set_ylabel(r'$|\iota_{N}|$', size = 22, color = 'g')

# color of axis

# ax.tick_params(axis='y', colors='r', labelsize = 14)
ax.tick_params(axis='y', colors='k', labelsize = 14)
ax.tick_params(axis='x', colors='k', labelsize = 14)
ax2.tick_params(axis='y', colors='b', labelsize = 14)
ax3.tick_params(axis='y', colors='g', labelsize = 14)

ax2.spines['right'].set_color('b')
ax3.spines['right'].set_color('g')
# ax3.spines['left'].set_color('r')
ax3.spines['left'].set_color('k')


ax2.set_ylim(np.min(eta_arr)-0.1, np.max(eta_arr)+0.1)

alpha_tilde_hold = alpha_tilde_arr[0]

ax2.text(a_arr[0]*2, np.min(eta_arr)*0.95, r'QA, $\tilde{\alpha} = 0$', fontsize = 20)

for idx, alpha_tile in enumerate(alpha_tilde_arr):

    if alpha_tilde_hold != alpha_tile:

        ax2.vlines(a_arr[idx], np.min(eta_arr)-1, np.max(eta_arr)+1, colors = 'k', linestyles = 'dashed', zorder = -1, alpha = 0.4)

        ax2.fill_between(a_arr[:idx+1], np.min(eta_arr)-1, np.max(eta_arr)+1, alpha=0.1, zorder = 0)

        ax2.text(a_arr[idx]*1.20, np.max(eta_arr)*1.4, r'QH, $\tilde{\alpha} = $' + ' {}'.format(int(alpha_tile)), fontsize = 20)

        alpha_tilde_hold = alpha_tile

ax.set_zorder(5)  # default zorder is 0 for ax1 and ax2
ax.patch.set_visible(False)  # prevents ax1 from hiding ax2

fig.tight_layout()
fig.savefig('nor_ae_scan_shape_axis_r_{}_{}_b_{}_{}.png'.format(r, nfp, b, N_a), format='png', dpi=300, bbox_inches='tight')
plt.show()

# MAYBE RUN OPTIMIZATION TO FIND THE BEST A AND ETA
# IN ANOTHER SCRIPT PROBABLY

idx_opt = np.argmin(ae_total_arr)
idx_opt_nor = np.argmin(ae_nor_total_arr)

print('for last gradient \n')
print('Optimal: \n a = {} \n eta = {}'.format(a_arr[idx_opt], eta_arr[idx_opt]))
print('Optimal nor: \n a = {} \n eta = {}'.format(a_arr[idx_opt_nor], eta_arr[idx_opt_nor]))