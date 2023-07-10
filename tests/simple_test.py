import sys
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py')
from ae_nae_first_order_routines import  *
from qsc import Qsc
import matplotlib.pyplot as plt
import timeit

# A configuration is defined with B0, etabar, field periods, Fourier components and eta bar

B0 = 1          # [T] strength of the magnetic field on axis

eta = -0.9      # parameter
eta = -1.551538720240993
nfp = 3         # number of field periods

rc=[1, 0.045]   # rc -> the cosine components of the axis radial component
zs=[0, -0.045]  # zs -> the sine components of the axis vertical component
rc=[1, 0.15864321608040202]   # rc -> the cosine components of the axis radial component
zs=[0, -0.15864321608040202]  # zs -> the sine components of the axis vertical component


#
# minimum of ae_total_arr: 0.00032931733186449875
# a for minimum of ae_total_arr: 0.02526315789473684
# b for minimum of ae_total_arr: -0.02210526315789474
# eta for minimum of ae_total_arr: -0.13021923885735884

# a for minimum of ae_total_arr: 0.01
# b for minimum of ae_total_arr: -0.01714285714285714
# eta for minimum of ae_total_arr: -0.29765947102889734


# 0.004350888535800758
# 0.1757142857142857
# -0.008571428571428563
# -0.3217907065743966


# 0.0020603768217659635
# 0.1757142857142857
# -0.008571428571428563
# -0.3217907065743966

#-1.5373688813367854

# minimum of ae_total_arr: 0.0003918763557246035
# a for minimum of ae_total_arr: 0.11357142857142856
# b for minimum of ae_total_arr: -0.06
# eta for minimum of ae_total_arr: -0.29765947102889734

eta = -0.29765947102889734
rc = [1, 0.11357142857142856, -0.06]
zs = [0, 0.11357142857142856, -0.06]


# minimum of ae_total_arr: 7.90990498818863e-05
# a for minimum of ae_total_arr: 0.105065
# b for minimum of ae_total_arr: -0.0552
# eta for minimum of ae_total_arr: -0.030262679452118205


# minimum of ae_total_arr: 5.101330701301146e-06
# a for minimum of ae_total_arr: 0.017094333333333333
# b for minimum of ae_total_arr: -0.0316
# eta for minimum of ae_total_arr: -0.005168700094514442

eta = -0.030262679452118205
rc = [1, 0.105065, -0.0552]
zs = [0, 0.105065, -0.0552]

eta = -0.9
rc = [1, 0.045]
zs = [0, 0.045]

# these quantities are provided in [m]

# this is the configuration of r1 section 5.1 from
# (not sure) Landreman, Sengupta, and Plunk, Journal of Plasma Physics 85, 905850103 (2019).

stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=eta, B0=B0)

# to calculate things the lam_res needs to be provided, just that?
lam_res = 5000

# distance from the magnetic axis to be used
r = 0.0005

# normalization variable for AE
Delta_psiA = 1

# gradients for diamagnetic frequency
dln_n_dpsi = -5
dln_T_dpsi = 0

#declaring the object to compute ae using anlaytical form

stel_ae = ae_nae(stel, r, lam_res, Delta_psiA)

#declaring the object to compute ae using numerical form

# vartheta_res = 10000

# stel_ae_num = ae_nae_num(stel, r, lam_res, vartheta_res, Delta_psiA)

# array with logaritmic spacing of r


r_arr = np.geomspace(1e-10, 0.1, 100)
ae_total_arr = np.empty_like(r_arr)

for idx, r in enumerate(r_arr):
    
    stel_ae = ae_nae(stel, r, lam_res, Delta_psiA)
    ae_total_arr[idx] = stel_ae.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)[1]
    #print("r = {}".format(r))
    #print("The time taken for analytical : {:.6f} s, AE total = {:.10f}".format(timeit.timeit(lambda: stel_ae.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)[1], number = 1), ae_total_arr[idx]))
    #print("\n")
# fig with specific size
fig = plt.figure(figsize=(7, 5))
plt.plot(r_arr, ae_total_arr, 'r*-', markersize = 1.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('r', fontsize = 14)
plt.ylabel('AE total', fontsize = 14)
plt.grid(alpha = 0.5)
plt.title('AE total vs r', fontsize = 16)
plt.show()

##### Test simple #######
# print("\n")
# print("r = {}".format(r))
# print("The time taken for analytical                        : {:.6f} s, AE total = {:.10f}".format(timeit.timeit(lambda: stel_ae.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)[1], number = 1), stel_ae.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)[1]))
# print("\n")
#print("The time taken for numerical, analytical z           : {:.6f} s, AE total = {:.10f}".format(timeit.timeit(lambda: stel_ae_num.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)[1], number = 1), stel_ae_num.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)[1]))
#print("The time taken for numerical, numerical  z and lambda: {:.6f} s, AE total = {:.10f}".format(timeit.timeit(lambda: stel_ae_num.NUM_ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)[1], number = 1), stel_ae_num.NUM_ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)[1]))

"""

# 1st plot comparison of bouncing times and precession frequencies
##################################################################

tau_b_num, w_alpha_num, w_psi_num = stel_ae_num.tau_b_w_alpha_w_psi()

lamb_arr_num = stel_ae_num.lam_arr

tau_b_anal = stel_ae.bounce_time_nor_analytical()
w_alpha_anal = stel_ae.w_alpha_nor_analytical()
lamb_arr_anal = stel_ae.lam_arr



fig,ax = plt.subplots(figsize=(11,5))
ax.grid(alpha = 0.5)

ax.plot(lamb_arr_anal, tau_b_anal, 'c', label = r'$\hat{\tau}_{b}$' + ' analytical')
ax.plot(lamb_arr_num, tau_b_num, '-.k', label = r'$\hat{\tau}_{b}$' + ' numerical')
ax.set_ylabel(r'$\hat{\tau}_{b}$')
ax.set_xlabel('$\lambda$')

ax.legend(loc = 3)

ax2=ax.twinx()

ax2.plot(lamb_arr_anal, w_alpha_anal, label = r'$\hat{\omega}_{\alpha}$' + ' analytical')
ax2.plot(lamb_arr_num, w_alpha_num, '-.', label = r'$\hat{\omega}_{\alpha}$' + ' numerical')

ax2.plot(lamb_arr_anal, 0*w_alpha_anal, label = r'$\hat{\omega}_{\psi}$' + ' analytical')
ax2.plot(lamb_arr_num, w_psi_num, '-.', label = r'$\hat{\omega}_{\psi}$' + ' numerical')

ax2.set_ylabel(r'$\hat{\omega}_{\psi}$' + ', ' + r'$\hat{\omega}_{\alpha}$')
ax2.legend(loc = 1)
ax2.set_title('Standard QA configuration \n' + r'$\vartheta_{res.}$' + '(for numerical) = {},   '.format(vartheta_res) + r'$\lambda_{res.}$' + '(for both) = {},  r = {}'.format(lam_res, r))

plt.show()


# 2nd plot, different total available energy results comparison for
# dln_n_dpsi
##################################################################

# array with different gradients
#omn_arr = np.linspace(-10, 10, 5)

omn_arr = [-10, -1, 1, 10]

fig,ax = plt.subplots(figsize=(11,5))

ax.set_xlabel('$\lambda$')
ax.set_ylabel('$\hat{A}_{\lambda}/\hat{A}$')
ax.grid(alpha = 0.5)
ax.set_title('Standard QA configuration \n' + r'$\vartheta_{res.}$' + '(for numerical) = {},   '.format(vartheta_res) + r'$\lambda_{res.}$' + '(for both) = {},  r = {}'.format(lam_res, r))

for idx, dln_n_dpsi in enumerate(omn_arr):

    ae_per_lam, ae_total = stel_ae.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)
    ae_per_lam_total = ae_per_lam/ae_total

    ax.plot(lamb_arr_anal, ae_per_lam_total, label = 'omn (anlaytical) = {:.2f}, AE = {:.4f}'.format(dln_n_dpsi, ae_total))


    ae_per_lam_num, ae_total_num = stel_ae_num.NUM_ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)
    ae_per_lam_total_num = ae_per_lam_num/ae_total_num

    ax.plot(lamb_arr_num, ae_per_lam_total_num, '-.', label = 'omn (numerical) = {:.2f}, AE = {:.4f}'.format(dln_n_dpsi, ae_total_num))

ax.legend()
ax.set_title('r = {}, eta = {}, omT = {}'.format(r, eta, dln_T_dpsi))

ax2=ax.twinx()

ax2.plot(lamb_arr_anal, w_alpha_anal, '-.k')
ax2.plot(lamb_arr_anal, 0*w_alpha_anal, '--k')
ax2.set_ylabel(r'$\hat{\omega}_{\alpha}$')

plt.show()


# 3rd plot, for an specific omn, graph the difference in the ralf style plot
############################################################################

# gradients for diamagnetic frequency
dln_n_dpsi = -5
dln_T_dpsi = 0


# setting vectors of B for plotting

iotaN = stel.iotaN

N = stel.iota - stel.iotaN

res = 5000

theta = np.linspace(-np.pi, np.pi, res)

phi = theta/(iotaN + N)

vartheta = theta - N*phi

magB = stel.B_mag(r=r, theta=theta, phi=phi)

vartheta_ae = 2*np.arcsin(np.sqrt((((1 - lamb_arr_anal*(1 + r*eta))/(-r*eta*lamb_arr_anal)))/2))

# calculating ae

ae_per_lam, ae_total = stel_ae.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)
ae_per_lam_num, ae_total_num = stel_ae_num.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)

# error
error_arr_plots = np.abs((ae_per_lam/ae_total) - (ae_per_lam_num/ae_total_num))

norm = plt.Normalize()
colors = plt.cm.plasma(norm(error_arr_plots))


cmap = plt.get_cmap('plasma', 200)
norm = mpl.colors.Normalize(vmin=np.min(error_arr_plots), vmax=np.max(error_arr_plots))
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


fig,ax = plt.subplots(figsize=(11,5))


for idx, var_theta in enumerate(vartheta_ae):

    B_vartheta = np.interp(var_theta, vartheta, magB)
    ax.plot([-var_theta, var_theta], [B_vartheta, B_vartheta], c = colors[idx], linewidth = 1.1)

ax.set_title('r = {} [m], $\eta$ = {}, B0 = {} [T], \n'.format(r, eta, B0) + r'$\Delta{\psi}_{A}$' + ' = {:.2f}'.format(1) + ', omT = {}, omn = {}'.format(dln_T_dpsi, dln_n_dpsi) + '\n AE analytical: {},  AE numerical: {} \n'.format(ae_total, ae_total_num) + r'$\vartheta_{res.}$' + '(for numerical) = {},   '.format(vartheta_res) + r'$\lambda_{res.}$' + '(for both) = {},  r = {}'.format(lam_res, r))
ax.plot(vartheta, magB, c = 'black')
fig.colorbar(sm, label = r'|$\hat{A}_{\lambda}/\hat{A}_{analytical}$ - $\hat{A}_{\lambda}/\hat{A}_{numerical}$|', location = 'left', pad = 0.16)
ax.set_ylabel('$|B|$')
ax.set_xlabel(r'$\vartheta$')

ax2=ax.twinx()

ax2.plot(vartheta_ae, w_alpha_anal, '-.k', linewidth = 0.9, alpha = 0.7)
ax2.plot(-vartheta_ae, w_alpha_anal, '-.k', linewidth = 0.9, alpha = 0.7)
ax2.plot([-vartheta_ae[0] - 0.1, vartheta_ae[0]+ 0.1], [0,0], '--k', linewidth = 0.5)
ax2.set_ylabel(r'$\hat{\omega}_{\alpha}$')

plt.show()

"""

###################################################################################


#stel_ae_num.plot_ae_per_lam(dln_n_dpsi, dln_T_dpsi)



#stel_ae.plot_ae_per_lam(dln_n_dpsi, dln_T_dpsi)
