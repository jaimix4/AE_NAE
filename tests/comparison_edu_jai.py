import sys
from qsc import Qsc
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_nor_func_pyqsc import ae_nor_nae
import numpy as np
# from AEpy import ae_routines as ae
import timeit

from   matplotlib        import rc
import matplotlib.pyplot as plt
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
rc('text', usetex=True)

names = ["precise QA","precise QA+well","precise QH","precise QH+well"]

# stel = Qsc.from_paper(name, nphi = nphi)
rc=[1, 0.045]
zs=[0, 0.045]
nfp=3
etabar=-0.9
B0=1

# stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=etabar, B0=B0)

stel = Qsc.from_paper('precise QH') #, nphi = nphi)
stel.spsi = -1
stel.zs = -stel.zs
# print(stel.spsi)
# print(stel.etabar)

r = 1e-3
lam_res = 10001
Delta_r = 1
a_r = 1
omn = 3*stel.spsi
omt = -2*stel.spsi

def ae_nor_total_function_eta(x):

    #return ae_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_psiA, omn, omT, plot = False)[-1]
    #print(omn)
    return ae_nor_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_r, a_r, omn, omt, plot = False)[-1]

ae_nae = ae_nor_nae(stel, r, lam_res, Delta_r, a_r, -omn, -omt, plot = True)[-1]

print(ae_nae)

# print("FT Volume is {}".format(ae_nae.Et))

# result from other code total AE is 0.037695300955572004