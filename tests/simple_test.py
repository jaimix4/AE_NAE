import sys
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py')
from ae_nae_first_order_routines import  *
from qsc import Qsc

# A configuration is defined with B0, etabar, field periods, Fourier components and eta bar

B0 = 1          # [T] strength of the magnetic field on axis

eta = -0.9      # parameter

nfp = 3         # number of field periods

rc=[1, 0.045]   # rc -> the cosine components of the axis radial component
zs=[0, -0.045]  # zs -> the sine components of the axis vertical component

# these quantities are provided in [m]

# this is the configuration of r1 section 5.1 from
# (not sure) Landreman, Sengupta, and Plunk, Journal of Plasma Physics 85, 905850103 (2019).

stel = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=nfp, etabar=eta, B0=B0)

# to calculate things the lam_res needs to be provided, just that?
lam_res = 10

r = 0.05

Delta_psiA = 1

dln_n_dpsi = -10

dln_T_dpsi = 0

stel_ae = ae_nae(stel, r, lam_res, Delta_psiA)

vartheta_res = 1000

stel_ae_num = ae_nae_num(stel, r, lam_res, vartheta_res, Delta_psiA)

print('analytical')
print(stel_ae.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)[1])

print('numerical')
print(stel_ae_num.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)[1])

# print(stel_ae.bounce_time_nor_analytical())
#


#
# print(stel_ae.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi))

# stel_ae.plot_ae_per_lam(dln_n_dpsi, dln_T_dpsi)
