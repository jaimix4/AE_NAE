from scipy.special import erf
from scipy.integrate import quad, quad_vec
from scipy.interpolate import interp1d
from BAD import bounce_int
import numpy as np
from qsc import Qsc
from AEpy import ae_routines as ae
from   matplotlib        import rc
import   matplotlib.pyplot  as plt
import sys
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_nor_func_pyqsc import ae_nor_nae
from opt_eta_star import set_eta_star
from qs_shape_opt import choose_Z_axis

rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)


# comparing same configuration 

# rc=[1, 0.045]
# zs=[0, 0.045]
# nfp=3
# etabar=-0.9
# B0=1

# nphi = int(1e3+1)
# stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=etabar, B0=B0, order = "r2", nphi=nphi)

# make stellarator in QSC
nphi = int(1e3+1)

# print(stel.etabar)

# set input variables
lam_res = 2001
omn = 3
omt = 2
omnigenous = False
nfp = 3
etabar = -0.5
B0 = 1
r = 1e-3

N = 20

# DO NOT CHANGE THIS!!
a_arr = np.linspace(0.0001, 0.3, N)
b_arr = np.zeros_like(a_arr)

a_z_arr = np.zeros_like(a_arr)
b_z_arr = np.zeros_like(a_arr)
eta_arr = np.zeros_like(a_arr)
B2c_arr = np.zeros_like(a_arr)
delB20_arr = np.zeros_like(a_arr)
rcrit_arr = np.zeros_like(a_arr)

try:

    a_z_arr = np.load('data_shape_a/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    b_z_arr = np.load('data_shape_a/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    eta_arr = np.load('data_shape_a/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    B2c_arr = np.load('data_shape_a/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    delB20_arr = np.load('data_shape_a/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    rcrit_arr = np.load('data_shape_a/rcrit_arr_N_{}_nfp_{}.npy'.format(N, nfp))

    ald_computed = True
    print("files found")

except:

    ald_computed = False
    print("files not found")

if ald_computed == False:

    for idx, a in enumerate(a_arr):

        b = 0

        rcOr=[1, a, b]
        zsOr=[0.0, a, b]

        stel = Qsc(rc=rcOr, zs=zsOr, B0 = B0, nfp=nfp, order='r1', nphi = 61)

        num_iters = choose_Z_axis(stel, max_num_iter=50)

        a_z_arr[idx] = stel.zs[1]
        b_z_arr[idx] = stel.zs[2]
        eta_arr[idx] = stel.etabar
        B2c_arr[idx] = stel.B2c
        delB20_arr[idx] = stel.B20_variation
        rcrit_arr[idx] = stel.r_singularity

        print("Number of iters: {}, for a = {}".format(num_iters, a))

np.save('data_shape_a/a_arr_N_{}_nfp_{}.npy'.format(N, nfp), a_arr)
np.save('data_shape_a/b_arr_N_{}_nfp_{}.npy'.format(N, nfp), b_arr)
np.save('data_shape_a/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp), a_z_arr)
np.save('data_shape_a/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp), b_z_arr)
np.save('data_shape_a/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp), eta_arr)
np.save('data_shape_a/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp), B2c_arr)
np.save('data_shape_a/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp), delB20_arr)
np.save('data_shape_a/rcrit_arr_N_{}_nfp_{}.npy'.format(N, nfp), rcrit_arr)


ae_nor_total_second_arr = np.empty_like(a_arr)
ae_nor_total_first_arr = np.empty_like(a_arr)


for idx, a in enumerate(a_arr):

    # #a = 0.05
    # rcOr=[1, a, b]
    # zsOr=[0.0, a, b]

    alpha = 1.0
    stel = Qsc(rc=[1, a_arr[idx], b_arr[idx]], zs=[1, a_z_arr[idx], b_z_arr[idx]], nfp=nfp, etabar=eta_arr[idx], B0=B0, order = "r2", B2c = B2c_arr[idx], nphi=nphi)
    stel.spsi = 1
    stel.calculate()
    # etabar_star = set_eta_star(stel)
    # stel.etabar = etabar_star
    # stel.spsi = -1
    # if a < 0.1:
    #     stel.zs = stel.zs
    # else:
    #     stel.zs = -stel.zs
    stel.r = r = 0.5e-2
    omn_input = omn
    omt_input = omt

    try:
        NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=alpha, N_turns=3, nphi=nphi,
                            lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
        NAE_AE.calc_AE(omn=stel.spsi*omn_input,omt=stel.spsi*omt_input,omnigenous=omnigenous)
        #NAE_AE.plot_AE_per_lam()
        if NAE_AE.ae_tot >= 0:
            NAE_AE.plot_AE_per_lam()
            ae_nor_total_second_arr[idx] = NAE_AE.ae_tot
            print("total AE is", NAE_AE.ae_tot)
        else:
            ae_nor_total_second_arr[idx] = NAE_AE.ae_tot
            print("Could not compute for a = ", a)
        

    except:

        ae_nor_total_second_arr[idx] =  NAE_AE.ae_tot
        print("Could not compute for a = ", a)
    

        # NAE_AE.plot_precession(nae=True,stel=stel,alpha=-alpha)
        # NAE_AE.plot_AE_per_lam()

    
    # except:
    #     print('Error with a = {}, eta = {}'.format(a, etabar_star)) 
    #     ae_nor_total_second_arr[idx] = np.nan


    # stel = Qsc(rc=[1, a], zs=[0, a], nfp=nfp, etabar=-0.5, B0=B0, order = "r1")
    # etabar_star = set_eta_star(stel)
    # stel.etabar = etabar_star
    # stel.spsi = -1
    # if a < 0.1:
    #     stel.zs = stel.zs
    # else:
    #     stel.zs = -stel.zs
    # stel.r = r = 1e-3
    ae_nor_total_first_arr[idx] = ae_nor_nae(stel, r, lam_res, 1, 1, omn, omt, plot = True)[-1]

    
# plot results
fig, ax = plt.subplots()

ax.plot(a_arr, ae_nor_total_first_arr, '-*', label = "AE first")
ax.plot(a_arr, ae_nor_total_second_arr, label = "AE second")
ax2 = ax.twinx()
ax2.plot(a_arr, eta_arr, '--r', label = "etabar")
# ax.set_yscale('log')
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$\hat{A}$")
ax2.set_ylabel(r"$r_{*}$")
ax.legend()
ax2.legend()
ax.set_title('r = {}, omn = {}, omt = {}'.format(r, omn, omt))
plt.savefig('a_scan_r_{}_N_{}_NEW.png'.format(r, N))
plt.show()