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

a = 0.045
b = 0 

# a = 0.23
# b = 0.0012


a = 0.0001
b = 0.009473684210526315

rcOr=[1, a, b]
zsOr=[0.0, a, b]

stel = Qsc(rc=rcOr, zs=zsOr, B0 = B0, nfp=nfp, order='r1', nphi = 61)
stel.spsi = -1
stel.calculate()
num_iters = choose_Z_axis(stel, max_num_iter=50)

stel_fo = Qsc(rc=rcOr, zs=stel.zs, etabar = stel.etabar, B0 = B0, nfp=nfp, order='r1', nphi = 61)
stel_fo.spsi = -1
stel_fo.calculate()

lam_res = 1001
omn = 3
omt = 2

nphi = int(1e3+1)
alpha = 1.0
omn_input = omn
omt_input = omt
omnigenous = False

N_r = 50
max_r = stel.r_singularity
print(max_r) # 0.0454464717191772
r_arr = np.geomspace(1e-4, 0.07, N_r)
ae_first = np.zeros_like(r_arr)
ae_second = np.zeros_like(r_arr)

for idx, r in enumerate(r_arr):

    ae_first[idx] = ae_nor_nae(stel_fo, r, lam_res, 1, 1, omn, omt, plot = False)[-1]
    stel.r = r
    NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=alpha, N_turns=3, nphi=nphi,
                            lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
    NAE_AE.calc_AE(omn=stel.spsi*omn_input,omt=stel.spsi*omt_input,omnigenous=omnigenous)
    ae_second[idx] = NAE_AE.ae_tot

stel.r = r_arr[-1]
NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=stel.r, alpha=alpha, N_turns=3, nphi=nphi,
                            lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
NAE_AE.calc_AE(omn=stel.spsi*omn_input,omt=stel.spsi*omt_input,omnigenous=omnigenous)
NAE_AE.plot_AE_per_lam()

delB20 = stel.B20_variation

print(ae_first)
print(ae_second)
# plot results
fig, ax = plt.subplots(figsize = (9,6))
ax.plot(r_arr, ae_second, label = "AE second")
ax.plot(r_arr, ae_first, label = "AE first")
#ax.vlines(max_r)
ax.set_xscale('log')
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$\hat{A}$")
ax.legend()
ax.set_title('Rc = {}, Zs = {}, omn = {}, omt = {}, dB20 = {}'.format(stel.rc, stel.zs, omn, omt, delB20))
plt.savefig('ae_comp_N_{}_{}_{}.png'.format(N_r, stel.rc[1], stel.rc[2]))
plt.show()


