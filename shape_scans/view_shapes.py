import numpy as np
import matplotlib.pyplot as plt
from qsc import Qsc
from matplotlib.widgets import Slider, Button
# import matplotlib as mpl
# mpl.rcParams['text.usetex'] = True
# from   matplotlib        import rc
# rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 12})
# rc('text', usetex=True)
#### defining curve

Nfp = 4

Rc = np.array([0.001, 0])

Zs = np.array([0.001, 0])

rc = np.array([1, 0.15864321608040202])

zs = np.array([0, 0.15864321608040202])

#### stel pyqsc

B0 = 1

eta = -1.5

# code to calculate r singularity, I need to assess this with Rogerio

# stel = Qsc(rc=rc, zs=zs, nfp=Nfp, etabar=eta, B0=B0)
# stel.plot()
# Qsc.calculate_r_singularity(stel)
# print(stel.r_singularity)

def R_func(Rc, phi, Nfp, gamma=None, a=None):

    R = 1

    for n, rc_n in enumerate(Rc):

        R = R + rc_n*np.cos((n+1)*Nfp*phi)

    if gamma is not None:

        return R + a*np.cos(gamma)

    else:

        return R

def Z_func(Zs, phi, Nfp, gamma=None, a=None):

    Z = 0

    for n, zs_n in enumerate(Zs):

        Z = Z + zs_n*np.sin((n+1)*Nfp*phi)

    if gamma is not None:

        return Z + a*np.sin(gamma)

    else:

        return Z


def cyl_to_cart(R, phi, Z):

    X = R*np.cos(phi)

    Y = R*np.sin(phi)

    return X, Y, Z



fig = plt.figure(figsize = (8,7))

plt.subplots_adjust(bottom=0.25)

ax = fig.add_subplot(projection='3d')

n = 100

phi = np.linspace(0, np.pi, n)
gamma = np.linspace(0, 2*np.pi, n)
phi, gamma = np.meshgrid(phi, gamma)

for a in [0.05, 0.15, 0.25]:

    R = R_func(Rc, phi, Nfp, gamma, a)

    Z = Z_func(Zs, phi, Nfp, gamma, a)

    x, y, z = cyl_to_cart(R, phi, Z)

    l = ax.plot_surface(x, y, z, alpha = 0.3)


a = 0.01

phi = np.linspace(0, 2*np.pi, n)
gamma = np.linspace(0, 2.*np.pi, n)


R = R_func(Rc, phi, Nfp, gamma, a)

Z = Z_func(Zs, phi, Nfp, gamma, a)

x, y, z = cyl_to_cart(R, phi, Z)

l = ax.plot(x, y, z, color = 'k', linewidth = 2.0, label = 'magnetic axis')


lim_z_up = 1
lim_z_do = -lim_z_up

#t = ax.plot(x, y, z, label='parametric curve')
ax.plot(np.zeros(3), np.zeros(3), np.linspace(-1.2, 1.2, 3), label = "axis of torus", linewidth = 2.0, linestyle='dashed')

ax.set_xlabel('x', size = 16)
ax.set_ylabel('y', size = 16)
ax.set_zlabel('z', size = 16)

ax.set_title('Rc = [1, {:.3f}], Zs = [0, {:.3f}],    Nfp = {}'.format(0.005, 0.005, Nfp), y=0.95, pad=-7, fontsize = 18, color = 'r')

ax.set_zlim(-1.3, 1.3)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)


ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True, shadow=True, fontsize = 18)
ax.set_axis_off()
plt.tight_layout()


aF = plt.axes([0.25, 0.03, 0.65, 0.03])
bF = plt.axes([0.25, 0.00, 0.65, 0.03])

aF = Slider(aF, 'a (axis shape: a)', -0.3, 0.3, 0.0, valstep = 0.0005)
# create slider for bF
bF = Slider(bF, 'b (axis shape: b)', -0.06, 0.06, 0.0, valstep = 0.0005)

def update(val):

    aF_use = aF.val
    bF_use = bF.val

    ax.clear()

    Rc = np.array([aF_use , bF_use])

    Zs = np.array([aF_use , bF_use])

    n = 100

    phi = np.linspace(0, np.pi, n)
    gamma = np.linspace(0, 2*np.pi, n)
    phi, gamma = np.meshgrid(phi, gamma)

    for a in [0.05, 0.15, 0.25]:

        R = R_func(Rc, phi, Nfp, gamma, a)

        Z = Z_func(Zs, phi, Nfp, gamma, a)

        x, y, z = cyl_to_cart(R, phi, Z)

        l = ax.plot_surface(x, y, z, alpha = 0.3)


    a = 0.01

    phi = np.linspace(0, 2*np.pi, n)
    gamma = np.linspace(0, 2*np.pi, n)

    R = R_func(Rc, phi, Nfp)

    Z = Z_func(Zs, phi, Nfp)

    x, y, z = cyl_to_cart(R, phi, Z)

    l = ax.plot(x, y, z, color = 'k', linewidth = 2.0, label = 'magnetic axis')


    ax.set_xlabel('x', size = 16)
    ax.set_ylabel('y', size = 16)
    ax.set_zlabel('z', size = 16)

    ax.set_title('Rc = [1, {:.3f}, {:.3f}], Zs = [0, {:.3f}, {:.3f}], Nfp = {}'.format(aF_use, bF_use, aF_use, bF_use, Nfp), y=0.95, pad=-2, fontsize = 18, color = 'r')

    ax.set_zlim(-1.3, 1.3)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)

    ax.plot(np.zeros(3), np.zeros(3), np.linspace(-1.2, 1.2, 3), label = "axis of torus", linewidth = 2.0, linestyle = 'dashed')

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True, shadow=True, fontsize = 18)

    ax.set_axis_off()

    plt.tight_layout()

    fig.canvas.draw_idle()


aF.on_changed(update)
bF.on_changed(update)



plt.show()
