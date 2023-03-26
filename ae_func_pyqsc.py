import sys
import numpy as np
from scipy.integrate import simpson
from scipy.special import ellipk, ellipe, erf, elliprj, elliprf
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import jit
from qsc import Qsc
import timeit

# functions to calculate elliptical integrals fasters
#####################################################
# this will go out of this main code, later

def ellip_pi_carlson(n,m):

    return  elliprf(0, 1-m, 1) + (n/3)*elliprj(0, 1 - m, 1, 1 - n)


@jit(nopython=True)
def my_ellip_k_e(k, tol=1e-10):
    a = np.ones_like(k)
    g = np.sqrt(1-k)
    #this a is square, but it just ones here, so
    # to be fast lets just leave like that
    c = np.sqrt(np.abs(a - g**2))
    n = 0
    sum_th = 0.5*c**2
    while True:
        a, g = (a + g) / 2, np.sqrt(a * g)
        c = (c**2)/(4*a) # old c with new a, perfect
        n = n + 1
        sum_th = sum_th + (2**(n - 1))*c**2

        if (np.abs(a - g) < tol).all():
            K = (np.pi/(2*a))

            return K, K*(1 - sum_th)

# warming up this function
som = my_ellip_k_e(np.array([0.5]))

def ae_nae(self, r, lam_res, Delta_psiA, dln_n_dpsi, dln_T_dpsi, ax = None, ax2 = None, cax = None, plot = False):

    # fast function to calculate ae, everything is normalize according
    # to "Available energy of trapped electrons in Miller tokamak equilibria" Ralf
    # this just applies to first order quasisymmetric configutions
    #  A GUIDE OR WHATEVER NEEDS TO BE ADDED TO KNOW EXACTLY WHAT IS BEING
    # CALCULATED, EVERYTHING IS NORMALIZED

    # grabbing parameters from the configuration object (stel)

    B0 = self.B0

    G0 = self.G0

    iota = self.iota

    iotaN = self.iotaN

    L = self.axis_length

    eta = self.etabar

    # it is not necessaty to exclude  the boundary values of lambda
    # but to keep consistent with nnumerical case, it is done here

    lamb_min = lamb_min = 1/(1 - r*eta)

    lamb_max = lamb_max = 1/(1 + r*eta)

    lam_arr = np.linspace(lamb_min, lamb_max, lam_res)[1:-1]

    # eliptic integrals values

    ellip_n = ((-1*(1 - (1 + r*eta)*lam_arr))/((1 + r*eta)*lam_arr))

    ellip_m = (-1*(1 - (1 + r*eta)*lam_arr))/(2*r*eta*lam_arr)

    ellip_K, ellip_E = my_ellip_k_e(ellip_m)

    ellip_Pi = ellip_pi_carlson(ellip_n, ellip_m)

    # calculating tau_nor

    a = np.abs(G0/iotaN)

    num_tau_nor = 2 * np.sqrt(2) * a * ellip_Pi

    den_tau_nor = (B0 + B0*r*eta) * np.sqrt(-1 * r * eta * lam_arr)

    tau_nor = num_tau_nor/den_tau_nor

    # calculating w_alpha_nor

    expr1 = 2 * eta * (1 + r*eta) * lam_arr * ellip_E

    expr2 = (-1 + (r*eta)**2) * eta * lam_arr * ellip_K

    expr3 = 2*r*(eta**2) * ellip_Pi

    num_dJnor_dpsi = a*np.sqrt(2)*(expr1 + expr2 - expr3)

    den_dJnor_dpsi = r*(-1 + r*eta)* (B0 + B0*r*eta)**2 * np.sqrt(-1 * r * eta * lam_arr)

    dJnor_dpsi = -2*num_dJnor_dpsi/den_dJnor_dpsi

    w_alpha_nor = (Delta_psiA*dJnor_dpsi)/tau_nor

    # calculting ae with analytical z integral, assuming omnigenous

    Ghat = tau_nor/L

    #############################
    # DEFINE: calculate c0 and c1

    c0 = (Delta_psiA * (dln_n_dpsi - 3/2 * dln_T_dpsi)) / w_alpha_nor
    c1 = 1.0 - (Delta_psiA * dln_T_dpsi) / w_alpha_nor

    condition1 = np.logical_and((c0>=0),(c1<=0))
    condition2 = np.logical_and((c0>=0),(c1>0))
    condition3 = np.logical_and((c0<0),(c1<0))

    ans = np.zeros(len(c1))

    ans[condition1]  = (2 * c0[condition1] - 5 * c1[condition1])
    ans[condition2]  = (2 * c0[condition2] - 5 * c1[condition2]) * erf(np.sqrt(c0[condition2]/c1[condition2])) + 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0[condition2] + 15 * c1[condition2] ) * np.sqrt(c0[condition2]/c1[condition2]) * np.exp( - c0[condition2]/c1[condition2] )
    ans[condition3]  = (2 * c0[condition3] - 5 * c1[condition3]) * (1 - erf(np.sqrt(c0[condition3]/c1[condition3]))) - 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0[condition3] + 15 * c1[condition3] ) * np.sqrt(c0[condition3]/c1[condition3]) * np.exp( - c0[condition3]/c1[condition3] )

    ae_per_lam = 3/16*ans*Ghat*w_alpha_nor**2

    ae_total = simpson(ae_per_lam, lam_arr)

    if ax is not None:

        plot = False

    if ax is None and plot == True:

        fig,ax = plt.subplots(figsize=(11,5))

    if ax is not None or plot == True:

        res = 5000

        theta = np.linspace(-np.pi, np.pi, res)

        N = iota - iotaN

        phi = theta/(iotaN + N)

        vartheta = theta - N*phi

        magB = self.B_mag(r=r, theta=theta, phi=phi)

        vartheta_ae = 2*np.arcsin(np.sqrt((((1 - lam_arr*(1 + r*eta))/(-r*eta*lam_arr)))/2))

        ae_per_lamb_nor_total_ae =  ae_per_lam/ae_total

        norm = plt.Normalize()
        colors = plt.cm.plasma(norm(ae_per_lamb_nor_total_ae))

        cmap = plt.get_cmap('plasma', 200)
        norm = mpl.colors.Normalize(vmin=np.min(ae_per_lamb_nor_total_ae), vmax=np.max(ae_per_lamb_nor_total_ae))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # mess I am doing
        #fig,ax = plt.subplots(figsize=(11,5))

        for idx, var_theta in enumerate(vartheta_ae):

            if vartheta[0] < vartheta[-1]:

                B_vartheta = np.interp(var_theta, vartheta, magB)

            else:

                B_vartheta = np.interp(var_theta, np.flip(vartheta), magB)
                
            ax.plot([-var_theta, var_theta], [B_vartheta, B_vartheta], c = colors[idx], linewidth = 0.9)

        ax.set_title('Analytical, r = {} [m], $\eta$ = {:.3f}, B0 = {} [T], \n'.format(r, eta, B0) + r'$\Delta{\psi}_{A}$' + ' = {:.2f}'.format(Delta_psiA) + ', omT = {}, omn = {}'.format(dln_T_dpsi, dln_n_dpsi) + ', total AE = {:.8f}'.format(ae_total))
        ax.plot(vartheta, magB, c = 'black')
        # mess I am doing

        if plot == True:

            fig.colorbar(sm, label = '$\hat{A}_{\lambda}/\hat{A}$', location = 'left', pad = 0.16)

        else:

            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("left", size="5%", pad=0.05)
            plt.colorbar(sm, cax=cax, label = '$\hat{A}_{\lambda}/\hat{A}$')

            #plt.colorbar(sm, ax = ax, label = '$\hat{A}_{\lambda}/\hat{A}$', location = 'left', pad = 0.16)

        #fig.colorbar(sm, label = '$\hat{A}_{\lambda}/\hat{A}$', location = 'left', pad = 0.16)
        #ax.colorbar(sm, label = '$\hat{A}_{\lambda}/\hat{A}$', location = 'left', pad = 0.16)
        ax.set_ylabel('$|B|$')
        ax.set_xlabel(r'$\vartheta$')

        if plot == True:

            ax2=ax.twinx()

        ax2.cla()
        ax2.clear()

        ax2.plot(vartheta_ae, w_alpha_nor, '-.r', linewidth = 0.9, alpha = 0.7)
        ax2.plot(-vartheta_ae, w_alpha_nor, '-.r', linewidth = 0.9, alpha = 0.7)
        ax2.plot([-vartheta_ae[0] - 0.1, vartheta_ae[0]+ 0.1], [0,0], '--r', linewidth = 0.5)

        ax2.set_ylabel(r'$\hat{\omega}_{\alpha}$', color = 'r')
        ax2.spines['right'].set_color('r')
        ax2.tick_params(colors='red', which='major')

        # mess I am doing
    if plot == True:

        plt.show()


    # mess I am doing
    return tau_nor, w_alpha_nor, ae_per_lam, ae_total







# trying the function

#stel = Qsc(rc=[1, 0.045], zs=[0, -0.045], nfp=3, etabar=-0.9, B0=1)
#ae_total = ae_nae(stel, r = 0.05, lam_res = 5000, Delta_psiA = 1, dln_n_dpsi = 5, dln_T_dpsi = 0)[3]
#print("The time taken for analytical: {:.6f} s, AE total = {:.10f}".format(timeit.timeit(lambda: ae_nae(stel, r = 0.05, lam_res = 5000, Delta_psiA = 1, dln_n_dpsi = 5, dln_T_dpsi = 0, plot = False), number = 1), ae_total))
