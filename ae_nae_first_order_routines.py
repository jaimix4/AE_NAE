import sys
import numpy as np
from scipy.integrate import quad, dblquad, simpson
from scipy.special import ellipk, ellipe, erf
from mpmath import ellippi
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/BAD-main/')
from BAD import bounce_int



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

        self.w_alpha = 'not computed yet'
        self.w_psi = 'not computed yet'
        self.tau_b = 'not computed yet'




    def tau_b_w_alpha_w_psi(self):

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

        return tau_b, w_alpha, w_psi



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

        ae_per_lam = 3/16*ans*Ghat*w_alpha**2

        ae_total = simpson(ae_per_lam, self.lam_arr)

        return ae_per_lam, ae_total

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

        self.lamb_min = lamb_min = 1/(1 - r*self.eta)

        self.lamb_max = lamb_max = 1/(1 + r*self.eta)

        self.lam_arr = np.linspace(self.lamb_min, self.lamb_max, lam_res)[1:-1]

        self.ellip_n = ((-1*(1 - (1 + r*self.eta)*self.lam_arr))/((1 + r*self.eta)*self.lam_arr))

        self.ellip_m = (-1*(1 - (1 + r*self.eta)*self.lam_arr))/(2*r*self.eta*self.lam_arr)

        self.tau_nor = 'not computed yet'

        self.w_alpha_nor = 'not computed yet'

        self.w_psi_nor = np.zeros_like(self.lam_arr)

        # geometry parameters analytical



    def bounce_time_nor_analytical(self):

        a = np.abs(self.G0/self.iotaN)

        num_tau_nor = 2 * np.sqrt(2) * a  * np.vectorize(ellippi, otypes=(float,))(self.ellip_n, self.ellip_m)

        den_tau_nor = (self.B0 + self.B0*self.r*self.eta) * np.sqrt(-1 * self.r * self.eta * self.lam_arr)

        tau_nor = num_tau_nor/den_tau_nor

        self.tau_nor = tau_nor

        return tau_nor


    def w_alpha_nor_analytical(self, lamb=None):

        #############################
        # DEFINE: dJnor/dpsi (normalization)

        if lamb == None:

            lamb = self.lam_arr

        r = self.r

        eta = self.eta

        ellip_m = self.ellip_m

        ellip_n = self.ellip_n

        a = np.abs(self.G0/self.iotaN)

        B0 = self.B0

        expr1 = 2 * eta * (1 + r*eta) * lamb * ellipe(ellip_m)

        expr2 = (-1 + (r*eta)**2) * eta * lamb * ellipk(ellip_m)

        expr3 = 2*r*(eta**2) * np.vectorize(ellippi, otypes=(float,))(ellip_n, ellip_m)

        num_dJnor_dpsi = a*np.sqrt(2)*(expr1 + expr2 - expr3)

        den_dJnor_dpsi = r*(-1 + r*eta)* (B0 + B0*r*eta)**2 * np.sqrt(-1 * r * eta * lamb)

        dJnor_dpsi = -2*num_dJnor_dpsi/den_dJnor_dpsi

        #############################
        # DEFINE: the normalized w_alpha (ralf normalization for AE, taking into account jaime normalization)

        # lets take out this 4, that I think is already outside in ralf things
        #w_alpha_nor = (4*nabla_psiA*dJnor_dpsi)/tau_nor
        w_alpha_nor = (self.Delta_psiA*dJnor_dpsi)/self.bounce_time_nor_analytical()

        self.w_alpha_nor = w_alpha_nor

        return w_alpha_nor


    def ae_integrand_per_lamb_total(self, dln_n_dpsi, dln_T_dpsi):

        #############################
        # variables to integrate

        # z -> normalized energy E/T0

        # lamb -> pitch angle mu*B0/E

        # they come already in the arrays o

        tau_nor = self.bounce_time_nor_analytical()

        w_alpha_nor = self.w_alpha_nor_analytical()

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

        ae_per_lam = 3/16*ans*Ghat*w_alpha_nor**2

        ae_total = simpson(ae_per_lam, self.lam_arr)

        return ae_per_lam, ae_total


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

            B_vartheta = np.interp(var_theta, vartheta, magB)
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
