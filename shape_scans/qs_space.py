import sys
import numpy as np
import matplotlib.pyplot as plt
from qsc import Qsc
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_func_pyqsc import ae_nae
from scipy.optimize import minimize
# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True

# from Rodriguez paper
# a goes from -0.3 to 0.3
a = 0.045

# b goes from -0.06 to 0.06
b = 0.000

rc = np.array([1, a, b])

zs = np.array([0, a, b])

nfp = 3

eta = -0.9

B0 = 1

lam_res = 5000

Delta_psiA = 1

omn = -5
omT = 0

r = 0.005

a_arr = np.linspace(0.01, 0.3, 200)
ae_total_arr = np.empty_like(a_arr)
eta_arr = np.empty_like(a_arr)
alpha_tilde_arr = np.empty_like(a_arr)

iota_arr = np.empty_like(a_arr)

eta_start = [-0.7]

# simple  minimazation, better for just optimizing eta


def ae_total_function_eta(x):

    return ae_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_psiA, omn, omT, plot = False)[-1]


for idx, a in enumerate(a_arr):

    rc = np.array([1, a])

    zs = np.array([0, a])

    result = minimize(ae_total_function_eta, eta_start, method='L-BFGS-B', bounds = [[-5, 1]])

    solution = result['x']

    eta_arr[idx] = solution[0]

    ae_total_arr[idx] = ae_total_function_eta(solution)

    stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta_arr[idx], B0=B0)

    alpha_tilde_arr[idx] = stel.iotaN - stel.iota

    iota_arr[idx] = np.abs(stel.G0/stel.iotaN)


fig,ax = plt.subplots(figsize=(11,6))
ax.plot(a_arr, ae_total_arr, '-*r', zorder = 3, linewidth = 2, markersize = 0.2)

ax.set_yscale('log')
ax.grid(alpha = 0.4)

ax.set_ylabel(r'$\hat{A}$', size = 22, color = 'r')
ax.set_xlabel('a (for axis shape)', size = 20)
ax.set_xlim(np.min(a_arr), np.max(a_arr))
ax.set_title('Scan over axis shape: Rc = [1, a], Zs = [0, a] \n' + r'$\bar{\eta} = \bar{\eta}_{*}$' + '(optimal), ' + ' B0 = {}, r = {}, omn = {}, omnT = {}, nfp = {}'.format(B0, r, omn, omT, nfp), size = 20)

# plotting second axis

ax2=ax.twinx()

ax3 = ax.twinx()

ax2.plot(a_arr, eta_arr, '-*b', zorder = 3, linewidth = 2, markersize = 0.2)

ax3.plot(a_arr, iota_arr, '-*g', zorder = 3, linewidth = 2, markersize = 0.2)

ax3.spines['right'].set_position(('axes', 1.13))

ax3.set_yscale('log')

ax2.set_ylabel(r'$\bar{\eta}_{*}$', size = 22, color = 'b')

ax3.set_ylabel(r'$|G0/\iota_N|$', size = 22, color = 'g')

# color of axis

ax.tick_params(axis='y', colors='r', labelsize = 14)
ax.tick_params(axis='x', colors='k', labelsize = 14)
ax2.tick_params(axis='y', colors='b', labelsize = 14)
ax3.tick_params(axis='y', colors='g', labelsize = 14)

ax2.spines['right'].set_color('b')
ax3.spines['right'].set_color('g')
ax3.spines['left'].set_color('r')


ax2.set_ylim(np.min(eta_arr)-0.1, np.max(eta_arr)+0.1)

alpha_tilde_hold = 0

ax2.text(a_arr[0]*2, np.min(eta_arr)*0.95, r'QA, $\tilde{\alpha} = 0$', fontsize = 20)

for idx, alpha_tile in enumerate(alpha_tilde_arr):

    if alpha_tilde_hold != alpha_tile:

        ax2.vlines(a_arr[idx], np.min(eta_arr)-1, np.max(eta_arr)+1, colors = 'k', linestyles = 'dashed', zorder = 0)

        ax2.fill_between(a_arr[:idx+1], np.min(eta_arr)-1, np.max(eta_arr)+1, alpha=0.1, zorder = 0)

        ax2.text(a_arr[idx]*1.20, np.max(eta_arr)*1.20, r'QH, $\tilde{\alpha} = $' + ' {}'.format(int(alpha_tile)), fontsize = 20)

        alpha_tilde_hold = alpha_tile


fig.tight_layout()
fig.savefig('ae_scan_shape_axis_a_{}_{}.png'.format(r, nfp), format='png', dpi=300, bbox_inches='tight')
plt.show()

idx_opt = np.argmin(ae_total_arr)

print('Optimal: \n a = {} \n eta = {}'.format(a_arr[idx_opt], eta_arr[idx_opt]))
