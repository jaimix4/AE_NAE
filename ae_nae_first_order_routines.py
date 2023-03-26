import sys
import numpy as np
from scipy.integrate import quad, quad_vec, dblquad, simpson
from scipy.special import ellipk, ellipe, erf, elliprj, elliprf
from mpmath import ellippi
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/BAD-main/')
from BAD import bounce_int
from numba import jit
#from elliptic_func import my_ellip_k_e,


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

# warm up my function
som = my_ellip_k_e(np.array([0.5]))

# FUNCTIONS FOR CALCULATING AE NUMERICALLY
###########################################

def w_diamag(dlnndx,dlnTdx,z):

    return ( dlnndx/z + dlnTdx * ( 1.0 - 3.0 / (2.0 * z) ) )


def AE_per_lam_per_z(walpha,wpsi,wdia,tau_b,z):
    r"""
    The available energy per lambda per z.
    BY RALF
    """
    geometry = ( wdia - walpha ) * walpha - wpsi**2 + np.sqrt( ( wdia - walpha )**2 + wpsi**2  ) * np.sqrt( walpha**2 + wpsi**2 )
    envelope = np.exp(-z) * np.power(z,5/2)
    jacobian = tau_b
    val      = geometry * envelope * jacobian
    return val/(4*np.sqrt(np.pi))

###########################################

class ae_nae_num:

    def __init__(self, stel, r, lam_res, vartheta_res, Delta_psiA):

        self.stel = stel

        self.eta = self.stel.etabar

        self.B0 = self.stel.B0

        self.r = r

        self.lam_res = lam_res

        self.vartheta_res = vartheta_res

        self.Delta_psiA = Delta_psiA

        # computed from pyqsc propeties

        self.L = self.stel.axis_length # axis length in meters

        self.iota = self.stel.iota

        self.iotaN = self.stel.iotaN

        self.N = self.iota - self.iotaN

        # straight field line coordinate

        self.theta = np.linspace(-np.pi, np.pi, self.vartheta_res)

        self.phi = self.theta/(self.iotaN + self.N)

        self.vartheta = self.theta - self.N*self.phi

        # some other things

        self.s_psi = np.sign(self.stel.Bbar) # what is this???

        # magnetic vectos

        # magnetic field vectorize
        self.Bmag = self.stel.B_mag(r=self.r, theta=self.theta, phi=self.phi)

        # modulus of magnetic field
        self.modb = self.Bmag/self.B0

        # grad_alpha
        self.grad_alpha = (self.s_psi/self.r)*(self.B0**2)*self.eta*np.cos(self.vartheta)

        # grad_psi
        self.grad_psi = self.r*(self.B0**3)*self.eta*np.sin(self.vartheta)

        # curv_alpha  = np.load('curv_alpha.npy')
        self.curv_alpha = (self.s_psi/self.r)*self.B0*self.eta*np.cos(self.vartheta)

        # curv_psi    = np.load('curv_psi.npy')
        # ??????? missing in document
        self.curv_psi = self.grad_psi/self.Bmag

        # jacobian
        I = 0
        self.jac = (self.stel.G0 + self.iota*I)/self.Bmag**2

        #dldvartheta !!!!!! I figure this out :))))
        self.dldtheta = np.abs((self.modb*self.jac)/self.iotaN)

        # make lambda array
        self.lam_arr = np.linspace(1/self.modb.max(),1/self.modb.min(),self.lam_res)[1:-1]

        self.w_alpha = None
        self.w_psi = None
        self.tau_b = None

        self.ae_per_lam = None
        self.ae_total = None



    def tau_b_w_alpha_w_psi(self):

        if self.w_alpha is None or self.w_psi is None or self.tau_b is None:

            gtrapz_arr_alpha    = []
            gtrapz_arr_psi      = []
            boundary_list = []

            w_alpha = np.empty_like(self.lam_arr)
            w_psi = np.empty_like(self.lam_arr)
            tau_b = np.empty_like(self.lam_arr)

            for idx, lam_val in enumerate(self.lam_arr):

                f = 1 - lam_val * self.modb
                tau_b_arr               = self.dldtheta
                bounce_time, roots      = bounce_int.bounce_integral_wrapper(f,tau_b_arr,self.vartheta,return_roots=True)
                alpha_arr               = self.dldtheta * ( lam_val * self.grad_alpha + 2 * ( 1/self.modb - lam_val ) * self.curv_alpha )
                num_arr_alpha           = bounce_int.bounce_integral_wrapper(f,alpha_arr,self.vartheta)
                psi_arr                 = self.dldtheta * ( lam_val * self.grad_psi   + 2 * ( 1/self.modb - lam_val ) * self.curv_psi )
                num_arr_psi             = bounce_int.bounce_integral_wrapper(f,psi_arr,self.vartheta)
                # check if roots cross boundary
                cross_per_lam           = []
                for idx2 in range(int(len(roots)/2)):
                    boundary_cross = roots[2*idx2] > roots[2*idx2+1]
                    cross_per_lam.append(boundary_cross)
                # make into list of lists
                boundary_list.append(np.asarray(cross_per_lam))
                gtrapz_arr_psi.append(np.asarray(num_arr_psi)/np.asarray(bounce_time))
                gtrapz_arr_alpha.append(np.asarray(num_arr_alpha)/np.asarray(bounce_time))

                w_alpha[idx] = num_arr_alpha[0]/bounce_time[0]
                w_psi[idx] = num_arr_psi[0]/bounce_time[0]

                tau_b[idx] = bounce_time[0]


            self.w_alpha = w_alpha
            self.w_psi = w_psi
            self.tau_b = tau_b

        return self.tau_b, self.w_alpha, self.w_psi



    def NUM_ae_integrand_per_lamb_total(self,omn,omt,omnigenous = False, fast = True):
        # loop over all lambda
        # BY RALF
        # Delta_x = self.Delta_x
        # Delta_y = self.Delta_y
        L_tot  = self.L #np.trapz(self.sqrtg*self.modb,self.theta)
        tau_b, w_alpha, w_psi = self.tau_b_w_alpha_w_psi()

        ae_at_lam_list = []

        if omnigenous==False:

            if fast == False:

                for lam_idx, lam_val in enumerate(self.lam_arr):
                    wpsi_at_lam     = w_psi[lam_idx]
                    walpha_at_lam   = w_alpha[lam_idx]
                    taub_at_lam     = tau_b[lam_idx]
                    integrand       = lambda x: AE_per_lam_per_z(walpha_at_lam,wpsi_at_lam,w_diamag(omn,omt,x),taub_at_lam,x)
                    ae_at_lam, _    = quad_vec(integrand,0.0,np.inf, epsrel=1e-6,epsabs=1e-20, limit=1000)
                    ae_at_lam_list.append(ae_at_lam/L_tot)
            else:

                for lam_idx, lam_val in enumerate(self.lam_arr):
                    wpsi_at_lam     = w_psi[lam_idx]
                    walpha_at_lam   = w_alpha[lam_idx]
                    taub_at_lam     = tau_b[lam_idx]
                    integrand       = lambda x: np.sum(AE_per_lam_per_z(walpha_at_lam,wpsi_at_lam,w_diamag(omn,omt,x),taub_at_lam,x))
                    ae_at_lam, _    = quad(integrand,0.0,np.inf, epsrel=1e-6,epsabs=1e-20, limit=1000)
                    ae_at_lam_list.append(ae_at_lam/L_tot)

        # if omnigenous==True:
        #     for lam_idx, lam_val in enumerate(self.lam):
        #         walpha_at_lam   = Delta_x*self.walpha[lam_idx]
        #         taub_at_lam     = self.taub[lam_idx]
        #         dlnndx          = -omn
        #         dlnTdx          = -omt
        #         c0 = Delta_x * (dlnndx - 3/2 * dlnTdx) / walpha_at_lam
        #         c1 = 1.0 - Delta_x * dlnTdx / walpha_at_lam
        #         ae_at_lam       = AE_per_lam(c0,c1,taub_at_lam,walpha_at_lam)
        #         ae_at_lam_list.append(ae_at_lam/L_tot)

        self.ae_per_lam = ae_at_lam_list

        # now do integral over lam to find total AE
        lam_arr   = np.asarray(self.lam_arr).flatten()
        ae_per_lam_summed = np.zeros_like(lam_arr)
        for lam_idx, lam_val in enumerate(lam_arr):
            ae_per_lam_summed[lam_idx] = np.sum(self.ae_per_lam[lam_idx])
        ae_tot = np.trapz(ae_per_lam_summed,lam_arr)
        # if self.normalize=='ft-vol':
        #     ae_tot = ae_tot/self.ft_vol
        self.ae_total = ae_tot

        return self.ae_per_lam, self.ae_total



    def ae_integrand_per_lamb_total(self, dln_n_dpsi, dln_T_dpsi):

        #############################
        # variables to integrate

        # z -> normalized energy E/T0

        # lamb -> pitch angle mu*B0/E

        # they come already in the arrays o

        tau_b, w_alpha, _ = self.tau_b_w_alpha_w_psi()


        #############################
        # DEFINE: G1/2nor (ralf variable taking into account jaime normalization of tau_b)

        Om = 1 # maybe this normalization is not doing well

        Ghat = tau_b/self.L

        #############################
        # DEFINE: calculate c0 and c1

        c0 = self.Delta_psiA * (dln_n_dpsi - 3/2 * dln_T_dpsi) / w_alpha
        c1 = 1.0 - self.Delta_psiA * dln_T_dpsi / w_alpha

        condition1 = np.logical_and((c0>=0),(c1<=0))
        condition2 = np.logical_and((c0>=0),(c1>0))
        condition3 = np.logical_and((c0<0),(c1<0))

        ans = np.zeros(len(c1))

        ans[condition1]  = (2 * c0[condition1] - 5 * c1[condition1])
        ans[condition2]  = (2 * c0[condition2] - 5 * c1[condition2]) * erf(np.sqrt(c0[condition2]/c1[condition2])) + 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0[condition2] + 15 * c1[condition2] ) * np.sqrt(c0[condition2]/c1[condition2]) * np.exp( - c0[condition2]/c1[condition2] )
        ans[condition3]  = (2 * c0[condition3] - 5 * c1[condition3]) * (1 - erf(np.sqrt(c0[condition3]/c1[condition3]))) - 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0[condition3] + 15 * c1[condition3] ) * np.sqrt(c0[condition3]/c1[condition3]) * np.exp( - c0[condition3]/c1[condition3] )

        ae_per_lam = (3/16)*ans*Ghat*w_alpha**2

        ae_total = simpson(ae_per_lam, self.lam_arr)

        self.ae_per_lam = ae_per_lam
        self.ae_total = ae_total

        return self.ae_per_lam, self.ae_total




    def plot_ae_per_lam(self, dln_n_dpsi, dln_T_dpsi):

        iotaN = self.iotaN

        N = self.iota - self.iotaN

        r = self.r

        eta = self.eta

        B0 = self.B0

        Delta_psiA = self.Delta_psiA

        lamb_arr = self.lam_arr

        res = 5000

        theta = np.linspace(-np.pi, np.pi, res)

        phi = theta/(iotaN + N)

        vartheta = theta - N*phi

        magB = self.stel.B_mag(r=r, theta=theta, phi=phi)

        vartheta_ae = 2*np.arcsin(np.sqrt((((1 - lamb_arr*(1 + r*eta))/(-r*eta*lamb_arr)))/2))

        ae_per_lamb, ae_total = self.NUM_ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)

        ae_per_lamb_nor_total_ae =  ae_per_lamb/ae_total

        norm = plt.Normalize()
        colors = plt.cm.plasma(norm(ae_per_lamb_nor_total_ae))

        cmap = plt.get_cmap('plasma', 200)
        norm = mpl.colors.Normalize(vmin=np.min(ae_per_lamb_nor_total_ae), vmax=np.max(ae_per_lamb_nor_total_ae))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig,ax = plt.subplots(figsize=(11,5))

        for idx, var_theta in enumerate(vartheta_ae):

            B_vartheta = np.interp(var_theta, vartheta, magB)
            ax.plot([-var_theta, var_theta], [B_vartheta, B_vartheta], c = colors[idx], linewidth = 0.9)

        ax.set_title('Numerical, r = {} [m], $\eta$ = {}, B0 = {} [T], \n'.format(r, eta, B0) + r'$\Delta{\psi}_{A}$' + ' = {:.2f}'.format(Delta_psiA) + ', omT = {}, omn = {}'.format(dln_T_dpsi, dln_n_dpsi) + ', total AE = {:.2f}'.format(ae_total))
        ax.plot(vartheta, magB, c = 'black')
        fig.colorbar(sm, label = '$\hat{A}_{\lambda}/\hat{A}$', location = 'left', pad = 0.16)
        ax.set_ylabel('$|B|$')
        ax.set_xlabel(r'$\vartheta$')

        ax2=ax.twinx()

        ax2.plot(vartheta_ae, self.w_alpha, '-.r', linewidth = 0.9, alpha = 0.7)
        ax2.plot(-vartheta_ae, self.w_alpha, '-.r', linewidth = 0.9, alpha = 0.7)
        ax2.plot([-vartheta_ae[0] - 0.1, vartheta_ae[0]+ 0.1], [0,0], '--r', linewidth = 0.5)

        ax2.set_ylabel(r'$\hat{\omega}_{\alpha}$', color = 'r')
        ax2.spines['right'].set_color('r')
        ax2.tick_params(colors='red', which='major')

        plt.show()




class ae_nae:

    def __init__(self, stel, r, lam_res, Delta_psiA):

        self.stel = stel

        self.lam_res = lam_res

        self.Delta_psiA = Delta_psiA

        self.r = r

        self.B0 = self.stel.B0

        self.eta = self.stel.etabar

        self.G0 = self.stel.G0

        self.iota = self.stel.iota

        self.iotaN = self.stel.iotaN

        self.L = self.stel.axis_length

        self.lamb_min  = 1/(1 - r*self.eta)

        self.lamb_max  = 1/(1 + r*self.eta)

        self.lam_arr = np.linspace(self.lamb_min, self.lamb_max, lam_res)[1:-1]

        self.ellip_n = ((-1*(1 - (1 + r*self.eta)*self.lam_arr))/((1 + r*self.eta)*self.lam_arr))

        self.ellip_m = (-1*(1 - (1 + r*self.eta)*self.lam_arr))/(2*r*self.eta*self.lam_arr)

        self.tau_nor = None

        self.w_alpha_nor = None

        self.w_psi_nor = np.zeros_like(self.lam_arr)

        self.ellip_Pi = ellip_pi_carlson(self.ellip_n, self.ellip_m)

        self.ae_per_lam = None

        self.ae_total = None



    def bounce_time_nor_analytical(self):

        if self.tau_nor is None:

            a = np.abs(self.G0/self.iotaN)

            if self.ellip_Pi is None:

                self.ellip_Pi = ellip_pi_carlson(self.ellip_n, self.ellip_m)

            num_tau_nor = 2 * np.sqrt(2) * a * self.ellip_Pi

            den_tau_nor = (self.B0 + self.B0*self.r*self.eta) * np.sqrt(-1 * self.r * self.eta * self.lam_arr)

            tau_nor = num_tau_nor/den_tau_nor

            self.tau_nor = tau_nor


        return self.tau_nor


    def w_alpha_nor_analytical(self, lamb=None):

        #############################
        # DEFINE: dJnor/dpsi (normalization)

        if self.w_alpha_nor is None:

            if lamb == None:

                lamb = self.lam_arr

            r = self.r

            eta = self.eta

            ellip_m = self.ellip_m

            ellip_n = self.ellip_n

            a = np.abs(self.G0/self.iotaN)

            B0 = self.B0

            ellip_K, ellip_E = my_ellip_k_e(ellip_m)

            #expr1 = 2 * eta * (1 + r*eta) * lamb * ellipe(ellip_m)

            #expr2 = (-1 + (r*eta)**2) * eta * lamb * ellipk(ellip_m)

            #expr3 = 2*r*(eta**2) * np.vectorize(ellippi, otypes=(float,))(ellip_n, ellip_m)

            expr1 = 2 * eta * (1 + r*eta) * lamb * ellip_E

            expr2 = (-1 + (r*eta)**2) * eta * lamb * ellip_K

            if self.ellip_Pi is None:

                self.ellip_Pi = ellip_pi_carlson(self.ellip_n, self.ellip_m)

            expr3 = 2*r*(eta**2) * self.ellip_Pi

            num_dJnor_dpsi = a*np.sqrt(2)*(expr1 + expr2 - expr3)

            den_dJnor_dpsi = r*(-1 + r*eta)* (B0 + B0*r*eta)**2 * np.sqrt(-1 * r * eta * lamb)

            dJnor_dpsi = -2*num_dJnor_dpsi/den_dJnor_dpsi

            #############################
            # DEFINE: the normalized w_alpha (ralf normalization for AE, taking into account jaime normalization)

            # lets take out this 4, that I think is already outside in ralf things
            #w_alpha_nor = (4*nabla_psiA*dJnor_dpsi)/tau_nor
            w_alpha_nor = (self.Delta_psiA*dJnor_dpsi)/self.bounce_time_nor_analytical()

            self.w_alpha_nor = w_alpha_nor


        return self.w_alpha_nor


    def ae_integrand_per_lamb_total(self, dln_n_dpsi, dln_T_dpsi):

        #############################
        # variables to integrate

        # z -> normalized energy E/T0

        # lamb -> pitch angle mu*B0/E

        # they come already in the arrays o

        #tau_nor =

        if self.ae_per_lam is None or self.ae_total is None:

            w_alpha_nor = self.w_alpha_nor_analytical()

            tau_nor = self.bounce_time_nor_analytical()

            #############################
            # DEFINE: G1/2nor (ralf variable taking into account jaime normalization of tau_b)

            Om = 1 # maybe this normalization is not doing well

            Ghat = tau_nor/self.L

            #############################
            # DEFINE: calculate c0 and c1

            # here is where you apply the minus signs of the w_alpha_nor?

            c0 = (self.Delta_psiA * (dln_n_dpsi - 3/2 * dln_T_dpsi)) / w_alpha_nor
            c1 = 1.0 - (self.Delta_psiA * dln_T_dpsi) / w_alpha_nor

            condition1 = np.logical_and((c0>=0),(c1<=0))
            condition2 = np.logical_and((c0>=0),(c1>0))
            condition3 = np.logical_and((c0<0),(c1<0))

            ans = np.zeros(len(c1))

            ans[condition1]  = (2 * c0[condition1] - 5 * c1[condition1])
            ans[condition2]  = (2 * c0[condition2] - 5 * c1[condition2]) * erf(np.sqrt(c0[condition2]/c1[condition2])) + 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0[condition2] + 15 * c1[condition2] ) * np.sqrt(c0[condition2]/c1[condition2]) * np.exp( - c0[condition2]/c1[condition2] )
            ans[condition3]  = (2 * c0[condition3] - 5 * c1[condition3]) * (1 - erf(np.sqrt(c0[condition3]/c1[condition3]))) - 2 / (3 *np.sqrt(np.pi)) * ( 4 * c0[condition3] + 15 * c1[condition3] ) * np.sqrt(c0[condition3]/c1[condition3]) * np.exp( - c0[condition3]/c1[condition3] )

            self.ae_per_lam = (3/16)*ans*Ghat*w_alpha_nor**2

            self.ae_total = simpson(self.ae_per_lam, self.lam_arr)

        return self.ae_per_lam, self.ae_total


    def plot_ae_per_lam(self, dln_n_dpsi, dln_T_dpsi):

        iotaN = self.iotaN

        N = self.iota - self.iotaN

        r = self.r

        eta = self.eta

        B0 = self.B0

        Delta_psiA = self.Delta_psiA

        lamb_arr = self.lam_arr

        res = 5000

        vartheta_ae = 2*np.arcsin(np.sqrt((((1 - lamb_arr*(1 + r*eta))/(-r*eta*lamb_arr)))/2))

        theta = np.linspace(-np.pi, np.pi, res)

        phi = theta/(iotaN + N)

        vartheta = theta - N*phi

        magB = self.stel.B_mag(r=r, theta=theta, phi=phi)

        ae_per_lamb, ae_total = self.ae_integrand_per_lamb_total(dln_n_dpsi, dln_T_dpsi)

        ae_per_lamb_nor_total_ae =  ae_per_lamb/ae_total

        norm = plt.Normalize()
        colors = plt.cm.plasma(norm(ae_per_lamb_nor_total_ae))

        cmap = plt.get_cmap('plasma', 200)
        norm = mpl.colors.Normalize(vmin=np.min(ae_per_lamb_nor_total_ae), vmax=np.max(ae_per_lamb_nor_total_ae))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        fig,ax = plt.subplots(figsize=(11,5))

        for idx, var_theta in enumerate(vartheta_ae):

            if vartheta[0] < vartheta[-1]:

                B_vartheta = np.interp(var_theta, vartheta, magB)

            else:

                B_vartheta = np.interp(var_theta, np.flip(vartheta), magB)

            ax.plot([-var_theta, var_theta], [B_vartheta, B_vartheta], c = colors[idx], linewidth = 0.9)

        ax.set_title('Analytical, r = {} [m], $\eta$ = {}, B0 = {} [T], \n'.format(r, eta, B0) + r'$\Delta{\psi}_{A}$' + ' = {:.2f}'.format(Delta_psiA) + ', omT = {}, omn = {}'.format(dln_T_dpsi, dln_n_dpsi) + ', total AE = {:.2f}'.format(ae_total))
        ax.plot(vartheta, magB, c = 'black')
        fig.colorbar(sm, label = '$\hat{A}_{\lambda}/\hat{A}$', location = 'left', pad = 0.16)
        ax.set_ylabel('$|B|$')
        ax.set_xlabel(r'$\vartheta$')

        ax2=ax.twinx()

        ax2.plot(vartheta_ae, self.w_alpha_nor, '-.r', linewidth = 0.9, alpha = 0.7)
        ax2.plot(-vartheta_ae, self.w_alpha_nor, '-.r', linewidth = 0.9, alpha = 0.7)
        ax2.plot([-vartheta_ae[0] - 0.1, vartheta_ae[0]+ 0.1], [0,0], '--r', linewidth = 0.5)

        ax2.set_ylabel(r'$\hat{\omega}_{\alpha}$', color = 'r')
        ax2.spines['right'].set_color('r')
        ax2.tick_params(colors='red', which='major')

        plt.show()
