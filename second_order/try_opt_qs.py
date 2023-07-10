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

# a = 0.045
# b = 0 

a = 0.23
b = 0.0012


rcOr=[1, a, b]
zsOr=[0.0, a, b]

stel = Qsc(rc=rcOr, zs=zsOr, B0 = B0, nfp=nfp, order='r1', nphi = 61)
stel.spsi = -1
stel.calculate()
num_iters = choose_Z_axis(stel, max_num_iter=50)

print(num_iters)
print(stel.order)
print(stel.rc)
print(stel.zs)
print(stel.etabar)
print(stel.B2c)
print(stel.B20_variation)
print(stel.spsi)
print(stel.r_singularity)

lam_res = 1001
omn = 3
omt = 2
r = 1e-4

stel_fo = Qsc(rc=rcOr, zs=stel.zs, etabar = stel.etabar, B0 = B0, nfp=nfp, order='r1', nphi = 61)
stel_fo.calculate()
print(stel_fo.order)
print(ae_nor_nae(stel_fo, r, lam_res, 1, 1, omn, omt, plot = True)[-1])

nphi = int(1e3+1)
alpha = 1.0
omn_input = omn
omt_input = omt
omnigenous = False

# stel.zs = -stel.zs
# stel.calculate()

NAE_AE = ae.AE_pyQSC(stel_obj = stel, r=r, alpha=alpha, N_turns=3, nphi=nphi,
                            lam_res=lam_res,get_drifts=True,normalize='ft-vol',AE_lengthscale='None')
NAE_AE.calc_AE(omn=stel.spsi*omn_input,omt=stel.spsi*omt_input,omnigenous=omnigenous)
NAE_AE.plot_AE_per_lam()
print(NAE_AE.ae_tot)