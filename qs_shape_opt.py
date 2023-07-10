"""
This module finds the parameters associated to the NAE following the optimised-QS prescription
"""

import logging
import numpy as np
from scipy import integrate as integ
from scipy import optimize
# from .util import mu0

mu0 = 4 * np.pi * 1e-7

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def choose_eta(self, criterion = "std"):
    """
    Compute the \eta parameter for a given axis according to some of the standard criteria. 
    The default criterion = "std" corresponds to the choice of \eta=\eta^*, the choice to 
    extremise the rotational transform. 
    """
    logger.debug('Calculating eta...')

    def iota_eta(x):
        self.etabar = x
        self.order = "r1"
        self.init_axis()
        self.solve_sigma_equation()
        # self.calculate() # It seems unnecessary having to recompute all these quantities just because some etabar was introduced in the calculate axis routine. Used to not use it but not properly updated
        if criterion == "std":
            val = -np.abs(self.iotaN)
        elif criterion == "curv_weight":
            self.r1_diagnostics()
            val = np.mean(self.elongation/self.curvature**2)
        elif criterion == "L_grad_B":
            self.r1_diagnostics()
            val = -np.mean(self.L_grad_B)
        return val

    # def jac(x):
    #     iota_eta(x)
    #     DMred = self.d_d_varphi[1:,1:] 
    #     if self.sigma0 == 0 and np.max(np.abs(self.rs)) == 0 and np.max(np.abs(self.zc)) == 0:
    #         # Case in which sigma is stellarator-symmetric:
    #         integSig = np.linalg.solve(DMred,self.sigma[1:])   # Invert differentiation matrix: as if first entry a zero, need to add it later
    #         integSig = np.insert(integSig,0,0)  # Add the first entry 0
    #         expSig = np.exp(2*self.iotaN*integSig)
    #         # d_phi_d_varphi = 1 + np.matmul(d_d_varphi,self.phi-self.varphi)
    #         return 2*np.abs(self.iotaN)/self.etabar*sum(expSig*(self.sigma**2-1+self.etabar**4/self.curvature**4))/sum(expSig*(self.sigma**2+1+self.etabar**4/self.curvature**4)) 
    #     else:
    #         # Case in which sigma is not stellarator-symmetric:
    #         avSig = sum(self.sigma*self.d_varphi_d_phi)/len(self.sigma)     # Separate the piece that gives secular part, so all things periodic
    #         integSigPer = np.linalg.solve(DMred,self.sigma[1:]-avSig)   # Invert differentiation matrix: as if first entry a zero, need to add it later
    #         integSig = integSigPer + avSig*self.varphi[1:]  # Include the secular piece
    #         integSig = np.insert(integSig,0,0)  # Add the first entry 0
    #         expSig_ext = np.append(np.exp(2*self.iotaN*integSig),np.exp(2*self.iotaN*(avSig*2*np.pi/self.nfp))) # Add endpoint at 2*pi for better integration
    #         integrand = self.sigma**2+self.etabar**4/self.curvature**4
    #         integrand = np.append(integrand,integrand[0])
    #         varphi_ext = np.append(self.varphi, 2 * np.pi / self.nfp)
    #         return 2*np.abs(self.iotaN)/self.etabar*integ.trapz(expSig_ext * (integrand-1), varphi_ext) \
    #             / integ.trapz(expSig_ext * (integrand + 1), varphi_ext)

    # opt = optimize.minimize(iota_eta, x0 = min(self.curvature), method='BFGS', jac = jac)

    opt = optimize.minimize_scalar(iota_eta, method = 'bounded', bounds = [-max(self.curvature), 0])
    # opt = optimize.minimize(iota_eta, x0 = min(self.curvature))

    self.etabar = opt.x
    self.calculate()
    return self.etabar #*np.sqrt(2/self.B0)

def choose_B22c(self, criterion = "std"):
    """
    Compute the B_{22}^C parameter for a given axis and \eta value according to some standard criteria. 
    The default criterion = "std" corresponds to the choice of B_{22}^C that minimises QS residual. 
    """
    logger.debug('Calculating B_22^C...')

    def B20dev(x):
        self.B2c = x
        self.order = "r2"
        if criterion == "std":
            self.calculate_r2()
            val = self.B20_variation
        elif criterion == "asp_ratio":
            self.calculate_r2()
            val = 1/self.r_singularity    
        return val 

    # Search for B_22^C in the interval bounded by B_22^C=+-20
    opt = optimize.minimize_scalar(B20dev, method = 'bounded', bounds = [3*self.B0*self.etabar**2/4-20/4*self.B0**4, \
        3*self.B0*self.etabar**2/4+20/4*self.B0**4])
    # opt = optimize.minimize(B20dev, x0 = 0)

    self.B2c = opt.x
    self.calculate_r2()
    B22c = -4*(self.B2c-3*self.B0*self.etabar**2/4)/self.B0**4
    if np.abs(np.abs(B22c)-20)<0.01:
        logger.warning("The search of B22c hit a bound.")

    return -4*(self.B2c-3*self.B0*self.etabar**2/4)/self.B0**4

def choose_Z_axis(self, num_harm = 0, max_num_iter = 70):
    """
    Search the Z harmonics of the axis that minimise the QS residual 
    """
    if num_harm == 0:
        num_harm = self.rc.size-1
    B0 = self.B0
    nfp = self.nfp
    rcOr = self.rc.copy()
    zsOr = self.zs.copy()

    def findZOpt(z, info):
        zs = zsOr.copy()
        for i in range(num_harm):
            zs[i+1] = z[i]*zsOr[i+1]

        self.zs = zs
        self.order = 'r0'
        self.calculate()

        # self.choose_eta(criterion="std")
        # self.choose_B22c(criterion="std")

        choose_eta(self, criterion="std")
        choose_B22c(self, criterion="std")
        #print(self.B20_variation)
        return self.B20_variation

    x0In = np.ones((num_harm,1))*0.9
    opt = optimize.minimize(findZOpt, x0 = x0In, args=({'Nfeval':0}), method = 'Nelder-Mead', \
        options={'xatol':0.003,'fatol': 0.01, 'maxiter': max_num_iter, 'maxfev': max_num_iter})
    N_iter = opt.nit
    zs=zsOr[1:num_harm+1]*opt.x #[0.0, a*opt.x[0], b*opt.x[1]]
    zs = np.insert(zs, [0], [0])
    self.zs = zs
    self.order = 'r0'
    self.calculate()
    # self.choose_eta(criterion="std")
    # self.choose_B22c(criterion="std")
    choose_eta(self, criterion="std")
    choose_B22c(self, criterion="std")
    self.order = "r2"
    self.calculate()

    return N_iter