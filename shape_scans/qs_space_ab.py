import sys
import numpy as np
import matplotlib.pyplot as plt
from qsc import Qsc
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_func_pyqsc import ae_nae
from scipy.optimize import minimize
import matplotlib as mpl
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

# minimum of ae_total_arr: 7.90990498818863e-05
# a for minimum of ae_total_arr: 0.105065
# b for minimum of ae_total_arr: -0.0552
# eta for minimum of ae_total_arr: -0.030262679452118205

lam_res = 5000

Delta_psiA = 1

omn = -5
omT = 0

r = 0.005

N = 301
a_arr = np.linspace(0.0001, 0.3, N)
b_arr = np.linspace(-0.06, 0.06, N)

# mesh grid a_arr and b_arr
a_mesh, b_mesh = np.meshgrid(a_arr, b_arr)

ae_total_arr = np.empty_like(a_mesh)
eta_arr = np.empty_like(a_mesh)

alpha_tilde_arr = np.empty_like(a_mesh)

iota_arr = np.empty_like(a_mesh)

eta_start = [-0.7]

# simple  minimazation, better for just optimizing eta

counter = 0
total_counter = len(a_arr)*len(b_arr)

def ae_total_function_eta(x):

    # Calculate the objective function for the ae_nae function
    # for a given value of etabar and a set of fixed parameters.
    
    return ae_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_psiA, omn, omT, plot = False)[-1]


for idx, a in enumerate(a_arr):
    
    for idx_b, b in enumerate(b_arr):

        rc = np.array([1, a, b])

        zs = np.array([0, a, b])

        result = minimize(ae_total_function_eta, eta_start, method='L-BFGS-B', bounds = [[-5, 1]], options={'disp': False})
    # results with out log output
    #     result = minimize(ae_total_function_eta, eta_start, method='L-BFGS-B', bounds = [[-5, 1]], options={'disp': False})
        solution = result['x']

        eta_arr[idx_b, idx] = solution[0]

        ae_total_arr[idx_b, idx] = ae_total_function_eta(solution)

        stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta_arr[idx_b, idx], B0=B0)

        alpha_tilde_arr[idx_b, idx] = stel.iotaN - stel.iota

        iota_arr[idx_b, idx] = np.abs(stel.G0/stel.iotaN)

        counter += 1

        if counter % 100 == 0:
            print("#############################################")
            print("\n configurations done: {}/{} \n ".format(counter, total_counter))
            print("#############################################")




# print the minimum of ae_total_arr
print("minimum of ae_total_arr: {}".format(np.min(ae_total_arr)))
# print the a and b for the minimum of ae_total_arr
print("a for minimum of ae_total_arr: {}".format(a_mesh[np.where(ae_total_arr == np.min(ae_total_arr))][0]))
print("b for minimum of ae_total_arr: {}".format(b_mesh[np.where(ae_total_arr == np.min(ae_total_arr))][0]))
# print the eta for the minimum of ae_total_arr
print("eta for minimum of ae_total_arr: {}".format(eta_arr[np.where(ae_total_arr == np.min(ae_total_arr))][0]))

# save arrays as npy files
np.save('ae_total_arr_N_{}.npy'.format(N), ae_total_arr)
np.save('eta_arr_N_{}.npy'.format(N), eta_arr)
np.save('alpha_tilde_arr_N_{}.npy'.format(N), alpha_tilde_arr)
np.save('iota_arr_N_{}.npy'.format(N), iota_arr)
# save mesh grids as npy files
np.save('a_mesh_N_{}.npy'.format(N), a_mesh)
np.save('b_mesh_N_{}.npy'.format(N), b_mesh)

# subplots 
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
# q: how to add a colorbar to each subplot?

cs = axs[0, 0].contourf(a_mesh, b_mesh, np.log10(ae_total_arr), 1000, cmap = mpl.colormaps['plasma'])

# plot in log scale
# cs = axs[0, 0].contourf(a_mesh, b_mesh, ae_total_arr, 100, norm = mpl.colors.LogNorm())

fig.colorbar(cs, ax=axs[0, 0], shrink=0.9)

axs[0, 0].set_ylabel('b')
axs[0, 0].set_title('log10(AE total)')
cs = axs[0, 1].contourf(a_mesh, b_mesh, eta_arr, 1000, cmap = mpl.colormaps['plasma'])
fig.colorbar(cs, ax=axs[0, 1], shrink=0.9)
axs[0, 1].set_title('eta')
cs = axs[1, 0].contourf(a_mesh, b_mesh, alpha_tilde_arr, 6, cmap = mpl.colormaps['plasma'])
fig.colorbar(cs, ax=axs[1, 0], shrink=0.9)
axs[1, 0].set_title('alpha tilde')
axs[1, 0].set_ylabel('b')
axs[1, 0].set_xlabel('a')
cs = axs[1, 1].contourf(a_mesh, b_mesh, np.log10(iota_arr), 1000, cmap = mpl.colormaps['plasma'])
fig.colorbar(cs, ax=axs[1, 1], shrink=0.9)
axs[1, 1].set_title('log10(|G0/iota|)')
axs[1, 1].set_xlabel('a')

# mark the minimum of ae_total_arr with a circle in red
axs[0, 0].plot(a_mesh[np.where(ae_total_arr == np.min(ae_total_arr))], b_mesh[np.where(ae_total_arr == np.min(ae_total_arr))], 'ro', alpha = 0.4)
axs[0, 1].plot(a_mesh[np.where(ae_total_arr == np.min(ae_total_arr))], b_mesh[np.where(ae_total_arr == np.min(ae_total_arr))], 'ro', alpha = 0.4)
axs[1, 0].plot(a_mesh[np.where(ae_total_arr == np.min(ae_total_arr))], b_mesh[np.where(ae_total_arr == np.min(ae_total_arr))], 'ro', alpha = 0.4)
axs[1, 1].plot(a_mesh[np.where(ae_total_arr == np.min(ae_total_arr))], b_mesh[np.where(ae_total_arr == np.min(ae_total_arr))], 'ro', alpha = 0.4)

fig.savefig('qs_ab_space_nfp{}_{}.png'.format(nfp, N), format='png', dpi=300, bbox_inches='tight')
plt.show()


