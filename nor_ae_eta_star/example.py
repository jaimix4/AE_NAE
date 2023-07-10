#import logging
import numpy as np
from scipy import integrate as integ
from scipy import optimize


from qsc import Qsc
import numpy as np
import time

a = 0.045
#b = 0.0012
b = 0
B0 = 1
nfp = 3
rcOr=[1, a, b]
zsOr=[0.0, a, b]
stel = Qsc(rc=rcOr, zs=zsOr, B0 = B0, nfp=nfp, order='r1', nphi = 61)

# Compute eta^* that extremises iota_bar_0
# Different criteria for choosing etabar can be coded (see the file in qs_optim_config.py)
#etabar = stel.choose_eta()




def choose_eta(self, criterion = "std"):
    """
    Compute the \eta parameter for a given axis according to some of the standard criteria. 
    The default criterion = "std" corresponds to the choice of \eta=\eta^*, the choice to 
    extremise the rotational transform. 
    """
    #logger.debug('Calculating eta...')

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

    opt = optimize.minimize_scalar(iota_eta, method = 'bounded', bounds = [0, max(self.curvature)])
    # opt = optimize.minimize(iota_eta, x0 = min(self.curvature))

    self.etabar = opt.x
    self.calculate()
    return self.etabar #*np.sqrt(2/self.B0)


etabar = choose_eta(stel)

print(etabar)
# Construct near-axis configuration
stel.etabar = etabar
stel.calculate()