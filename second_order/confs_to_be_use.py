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
import matplotlib.gridspec as gridspec
from   matplotlib        import rc
import matplotlib.ticker as ticker
# add latex fonts to plots
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 18})
rc('text', usetex=True)

# set configuration space
B0 = 1
nfp = 3


N = 150
# NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
a_arr = np.linspace(0.0001, 0.3, N)
b_arr = np.linspace(-0.06, 0.06, N)

nfp = 3

# NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
a_arr = np.linspace(0.0001, 0.3, N)
b_arr = np.linspace(-0.06, 0.06, N)

# nfp = 2

# # NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
# a_arr = np.linspace(0.0001, 0.6, N)
# b_arr = np.linspace(-0.13, 0.13, N)

# nfp = 4

# # NEVER CHANGE THIS VALUES, AT LEAST FOR NFP = 3
# a_arr = np.linspace(0.0001, 0.20, N)
# b_arr = np.linspace(-0.0347, 0.0347, N)

# mesh grid a_arr and b_arr
a_mesh, b_mesh = np.meshgrid(a_arr, b_arr)



try:
    a_mesh = np.load('shapes_2nd/a_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    b_mesh = np.load('shapes_2nd/b_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    a_z_arr = np.load('shapes_2nd/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    b_z_arr = np.load('shapes_2nd/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    eta_arr = np.load('shapes_2nd/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    B2c_arr = np.load('shapes_2nd/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    delB20_arr = np.load('shapes_2nd/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp))
    rcrit_arr = np.load('shapes_2nd/rcrit_arr_N_{}_nfp_{}.npy'.format(N, nfp))

    ald_computed = True
    print("files found")

except:

    ald_computed = False
    print("files not found")



# # make a figure of 3 figures in a row
# fig = plt.figure(figsize = (7/1.5, 6/1.5))

##################### delB20 #####################

fig = plt.figure(figsize=(22/1.5, 7.4/1.5))
grid = gridspec.GridSpec(2, 3, wspace=0.3, hspace = 0.54, height_ratios=[1, 0.05])


# fig, axs = plt.subplots(1, 3, figsize = (21/1.5, 6/1.5))

nfp_arr = [2,3,4]

for idx, nfp in enumerate(nfp_arr):

    N = 150
    nfp = nfp_arr[idx]

    ax = plt.subplot(grid[0, idx])

    try:
        a_mesh = np.load('shapes_2nd/a_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        b_mesh = np.load('shapes_2nd/b_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        a_z_arr = np.load('shapes_2nd/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        b_z_arr = np.load('shapes_2nd/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        eta_arr = np.load('shapes_2nd/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        B2c_arr = np.load('shapes_2nd/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        delB20_arr = np.load('shapes_2nd/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        rcrit_arr = np.load('shapes_2nd/rcrit_arr_N_{}_nfp_{}.npy'.format(N, nfp))

        ald_computed = True
        print("files found")

    except:

        ald_computed = False
        print("files not found")


    # levels = np.linspace(np.log10(np.min(delB20_arr)), np.log10(np.max(delB20_arr)), 300)
    levels = np.linspace(-3.5, 9, 300)
    # levels = np.linspace(np.log10(np.min(delB20_arr)), 6, 200)
    cs = ax.contourf(a_mesh, b_mesh, np.log10(delB20_arr), levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')

    # if nfp == 4:
    #     fig.colorbar(cs, shrink=0.9)

    #     cax = fig.add_subplot(ax)  # Colorbar subplot
    #     cbar = fig.colorbar(cs, cax=cax)

    ax.set_title(r'$\log_{10} \: \Delta B_{20}$', fontsize = 24)

    if nfp == 2:

        a_arr = np.linspace(0.0001, 0.6, N)
        line_1 = lambda a: (-0.0612/0.22)*a - 0.0612
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.1171 + 0.0612)/0.6)*a - 0.0612
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.107, 0.269, 0.446, 0.512, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0584, 0.0588, 0.0604, 0.0695, 0.0898, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.45, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.6)
        ax.set_ylim(-0.13, 0.13)


    elif nfp == 3:

        a_arr = np.linspace(0.0001, 0.3, N)
        line_1 = lambda a: (-0.0280/0.108)*a - 0.0280
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0538 + 0.0280)/0.3)*a - 0.0280
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0591, 0.1433, 0.2211, 0.2677, 0.3] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0267, 0.0267, 0.0285, 0.0319, 0.0437, 0.0526] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.2285, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.3)
        ax.set_ylim(-0.06, 0.06)


    elif nfp == 4:

        a_arr = np.linspace(0.0001, 0.20, N)
        line_1 = lambda a: (-0.0160/0.064)*a - 0.0160
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0338 + 0.0160)/0.1891)*a - 0.0160
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0381, 0.0724, 0.1345, 0.1659, 0.1927] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0146, 0.0155, 0.0157, 0.0191, 0.0273, 0.0342] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.1368, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)


        ax.set_xlim(0.0001, 0.20)
        ax.set_ylim(-0.0347, 0.0347)


    # add text to figures
    ax.text(0.7, 0.05, r'$N_{\rm fp}$' + ' = {}'.format(nfp), fontsize = 18, transform=ax.transAxes, color = 'black') 


    ax.text(0.02, 0.420, r'${\rm QA}($' + '{})'.format(nfp*0), fontsize = 14, transform=ax.transAxes) 

    ax.text(0.1, 0.85, r'${\rm QH}($' + '{})'.format(nfp*2), fontsize = 14, transform=ax.transAxes) 

    ax.text(0.02, 0.02, r'${\rm QH}($' + '{})'.format(nfp*2), fontsize = 14, transform=ax.transAxes) 

    ax.text(0.75, 0.40, r'${\rm QH}($' + '{})'.format(nfp), fontsize = 14, transform=ax.transAxes) 

    # set number of decimal places in axis ticks
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    ax.plot(a_mesh[(26, 98)], b_mesh[(26, 98)], '*', color = 'gray', markersize = 25)

    
    if idx == 2:
        ax.set_ylabel(r'$b$', fontsize = 28)
        # put y label on the right
        ax.yaxis.set_label_position("right")

    ax.set_xlabel(r'$a$', fontsize = 28)
    

# cax = fig.add_axes([0.92, 0.14, 0.023, 0.7])  # [left, bottom, width, height]
# cbar = fig.colorbar(cs, cax=cax, orientation='vertical')

cax = fig.add_subplot(grid[1, :])  # Colorbar subplot for plots 1 and 2
cbar = fig.colorbar(cs, cax=cax, orientation='horizontal')
cbar.ax.locator_params(nbins=8)

# Adjust the size of color bar ticks
cbar.ax.tick_params(labelsize=20, direction='out')
cbar.ax.xaxis.set_label_position('top')


plt.savefig('configs_to_be_use_delB20.png', format='png', dpi=300, bbox_inches='tight')
plt.show()



##################### eta #####################

fig = plt.figure(figsize=(22/1.5, 7.4/1.5))
grid = gridspec.GridSpec(2, 3, wspace=0.3, hspace = 0.54, height_ratios=[1, 0.05])


# fig, axs = plt.subplots(1, 3, figsize = (21/1.5, 6/1.5))

nfp_arr = [2,3,4]

for idx, nfp in enumerate(nfp_arr):

    N = 150
    nfp = nfp_arr[idx]

    ax = plt.subplot(grid[0, idx])

    try:
        a_mesh = np.load('shapes_2nd/a_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        b_mesh = np.load('shapes_2nd/b_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        a_z_arr = np.load('shapes_2nd/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        b_z_arr = np.load('shapes_2nd/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        eta_arr = np.load('shapes_2nd/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        B2c_arr = np.load('shapes_2nd/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        delB20_arr = np.load('shapes_2nd/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        rcrit_arr = np.load('shapes_2nd/rcrit_arr_N_{}_nfp_{}.npy'.format(N, nfp))

        ald_computed = True
        print("files found")

    except:

        ald_computed = False
        print("files not found")


    # levels = np.linspace(np.log10(np.min(delB20_arr)), np.log10(np.max(delB20_arr)), 300)
    #levels = np.linspace(-3.5, 9, 300)
    # levels = np.linspace(np.log10(np.min(delB20_arr)), 6, 200)
    levels = np.linspace(-2.3, -0.01, 300)
    cs = plt.contourf(a_mesh, b_mesh, eta_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')

    # if nfp == 4:
    #     fig.colorbar(cs, shrink=0.9)

    #     cax = fig.add_subplot(ax)  # Colorbar subplot
    #     cbar = fig.colorbar(cs, cax=cax)

    ax.set_title(r'$N_{\rm fp}$' + ' = {},      '.format(nfp) + r'$\quad \bar{\eta}$', fontsize = 24)

    #ax.set_title(r'$\log_{10} \: \Delta B_{20}$', fontsize = 18)

    if nfp == 2:

        a_arr = np.linspace(0.0001, 0.6, N)
        line_1 = lambda a: (-0.0612/0.22)*a - 0.0612
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.1171 + 0.0612)/0.6)*a - 0.0612
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.107, 0.269, 0.446, 0.512, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0584, 0.0588, 0.0604, 0.0695, 0.0898, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.45, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.6)
        ax.set_ylim(-0.13, 0.13)


    elif nfp == 3:

        a_arr = np.linspace(0.0001, 0.3, N)
        line_1 = lambda a: (-0.0280/0.108)*a - 0.0280
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0538 + 0.0280)/0.3)*a - 0.0280
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0591, 0.1433, 0.2211, 0.2677, 0.3] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0267, 0.0267, 0.0285, 0.0319, 0.0437, 0.0526] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.2285, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.3)
        ax.set_ylim(-0.06, 0.06)


    elif nfp == 4:

        a_arr = np.linspace(0.0001, 0.20, N)
        line_1 = lambda a: (-0.0160/0.064)*a - 0.0160
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0338 + 0.0160)/0.1891)*a - 0.0160
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0381, 0.0724, 0.1345, 0.1659, 0.1927] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0146, 0.0155, 0.0157, 0.0191, 0.0273, 0.0342] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.1368, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)


        ax.set_xlim(0.0001, 0.20)
        ax.set_ylim(-0.0347, 0.0347)


    # add text to figures
    # ax.text(0.7, 0.05, r'$N_{\rm fp}$' + ' = {}'.format(nfp), fontsize = 18, transform=ax.transAxes, color = 'black') 


    # ax.text(0.02, 0.420, r'${\rm QA}($' + '{})'.format(nfp*0), fontsize = 14, transform=ax.transAxes) 

    # ax.text(0.1, 0.85, r'${\rm QH}($' + '{})'.format(nfp*2), fontsize = 14, transform=ax.transAxes) 

    # ax.text(0.02, 0.02, r'${\rm QH}($' + '{})'.format(nfp*2), fontsize = 14, transform=ax.transAxes) 

    # ax.text(0.75, 0.40, r'${\rm QH}($' + '{})'.format(nfp), fontsize = 14, transform=ax.transAxes) 

    # set number of decimal places in axis ticks
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    if idx == 2:
        ax.set_ylabel(r'$b$', fontsize = 28)
        # put y label on the right
        ax.yaxis.set_label_position("right")

    ax.set_xlabel(r'$a$', fontsize = 28)
    

# cax = fig.add_axes([0.92, 0.14, 0.023, 0.7])  # [left, bottom, width, height]
# cbar = fig.colorbar(cs, cax=cax, orientation='vertical')

cax = fig.add_subplot(grid[1, :])  # Colorbar subplot for plots 1 and 2
cbar = fig.colorbar(cs, cax=cax, orientation='horizontal')
cbar.ax.locator_params(nbins=8)

plt.savefig('configs_to_be_use_eta.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

##################### B2c #####################

fig = plt.figure(figsize=(22/1.5, 7.4/1.5))
grid = gridspec.GridSpec(2, 3, wspace=0.3, hspace = 0.54, height_ratios=[1, 0.05])


# fig, axs = plt.subplots(1, 3, figsize = (21/1.5, 6/1.5))

nfp_arr = [2,3,4]

for idx, nfp in enumerate(nfp_arr):

    N = 150
    nfp = nfp_arr[idx]

    ax = plt.subplot(grid[0, idx])

    try:
        a_mesh = np.load('shapes_2nd/a_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        b_mesh = np.load('shapes_2nd/b_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        a_z_arr = np.load('shapes_2nd/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        b_z_arr = np.load('shapes_2nd/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        eta_arr = np.load('shapes_2nd/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        B2c_arr = np.load('shapes_2nd/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        delB20_arr = np.load('shapes_2nd/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        rcrit_arr = np.load('shapes_2nd/rcrit_arr_N_{}_nfp_{}.npy'.format(N, nfp))

        ald_computed = True
        print("files found")

    except:

        ald_computed = False
        print("files not found")


    # levels = np.linspace(np.log10(np.min(delB20_arr)), np.log10(np.max(delB20_arr)), 300)
    #levels = np.linspace(-3.5, 9, 300)
    # levels = np.linspace(np.log10(np.min(delB20_arr)), 6, 200)
    levels = np.linspace(-4.2, 8.5, 300)
    cs = plt.contourf(a_mesh, b_mesh, B2c_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')

    # if nfp == 4:
    #     fig.colorbar(cs, shrink=0.9)

    #     cax = fig.add_subplot(ax)  # Colorbar subplot
    #     cbar = fig.colorbar(cs, cax=cax)

    #ax.set_title('nfp = {},      '.format(nfp) + r'$B_{2c}$', fontsize = 24)

    ax.set_title(r'$N_{\rm fp}$' + ' = {},      '.format(nfp) + r'$\quad B_{2c}$', fontsize = 24)

    #ax.set_title(r'$\log_{10} \: \Delta B_{20}$', fontsize = 18)

    if nfp == 2:

        a_arr = np.linspace(0.0001, 0.6, N)
        line_1 = lambda a: (-0.0612/0.22)*a - 0.0612
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.1171 + 0.0612)/0.6)*a - 0.0612
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.107, 0.269, 0.446, 0.512, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0584, 0.0588, 0.0604, 0.0695, 0.0898, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.45, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.6)
        ax.set_ylim(-0.13, 0.13)


    elif nfp == 3:

        a_arr = np.linspace(0.0001, 0.3, N)
        line_1 = lambda a: (-0.0280/0.108)*a - 0.0280
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0538 + 0.0280)/0.3)*a - 0.0280
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0591, 0.1433, 0.2211, 0.2677, 0.3] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0267, 0.0267, 0.0285, 0.0319, 0.0437, 0.0526] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.2285, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.3)
        ax.set_ylim(-0.06, 0.06)


    elif nfp == 4:

        a_arr = np.linspace(0.0001, 0.20, N)
        line_1 = lambda a: (-0.0160/0.064)*a - 0.0160
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0338 + 0.0160)/0.1891)*a - 0.0160
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0381, 0.0724, 0.1345, 0.1659, 0.1927] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0146, 0.0155, 0.0157, 0.0191, 0.0273, 0.0342] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.1368, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)


        ax.set_xlim(0.0001, 0.20)
        ax.set_ylim(-0.0347, 0.0347)


    # add text to figures
    # ax.text(0.7, 0.05, r'$N_{\rm fp}$' + ' = {}'.format(nfp), fontsize = 18, transform=ax.transAxes, color = 'black') 


    # ax.text(0.02, 0.420, r'${\rm QA}($' + '{})'.format(nfp*0), fontsize = 14, transform=ax.transAxes) 

    # ax.text(0.1, 0.85, r'${\rm QH}($' + '{})'.format(nfp*2), fontsize = 14, transform=ax.transAxes) 

    # ax.text(0.02, 0.02, r'${\rm QH}($' + '{})'.format(nfp*2), fontsize = 14, transform=ax.transAxes) 

    # ax.text(0.75, 0.40, r'${\rm QH}($' + '{})'.format(nfp), fontsize = 14, transform=ax.transAxes) 

    # set number of decimal places in axis ticks
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    if idx == 2:
        ax.set_ylabel(r'$b$', fontsize = 28)
        # put y label on the right
        ax.yaxis.set_label_position("right")

    ax.set_xlabel(r'$a$', fontsize = 28)
    

# cax = fig.add_axes([0.92, 0.14, 0.023, 0.7])  # [left, bottom, width, height]
# cbar = fig.colorbar(cs, cax=cax, orientation='vertical')

cax = fig.add_subplot(grid[1, :])  # Colorbar subplot for plots 1 and 2
cbar = fig.colorbar(cs, cax=cax, orientation='horizontal')
cbar.ax.locator_params(nbins=8)

plt.savefig('configs_to_be_use_B20.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

##################### r_crit #####################

fig = plt.figure(figsize=(22/1.5, 7.6/1.5))
grid = gridspec.GridSpec(2, 3, wspace=0.3, hspace = 0.54, height_ratios=[1, 0.05])


# fig, axs = plt.subplots(1, 3, figsize = (21/1.5, 6/1.5))

nfp_arr = [2,3,4]

for idx, nfp in enumerate(nfp_arr):

    N = 150
    nfp = nfp_arr[idx]

    ax = plt.subplot(grid[0, idx])

    try:
        a_mesh = np.load('shapes_2nd/a_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        b_mesh = np.load('shapes_2nd/b_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        a_z_arr = np.load('shapes_2nd/a_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        b_z_arr = np.load('shapes_2nd/b_z_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        eta_arr = np.load('shapes_2nd/eta_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        B2c_arr = np.load('shapes_2nd/B2c_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        delB20_arr = np.load('shapes_2nd/delB20_arr_N_{}_nfp_{}.npy'.format(N, nfp))
        rcrit_arr = np.load('shapes_2nd/rcrit_arr_N_{}_nfp_{}.npy'.format(N, nfp))

        ald_computed = True
        print("files found")

    except:

        ald_computed = False
        print("files not found")


    # levels = np.linspace(np.log10(np.min(delB20_arr)), np.log10(np.max(delB20_arr)), 300)
    #levels = np.linspace(-3.5, 9, 300)
    # levels = np.linspace(np.log10(np.min(delB20_arr)), 6, 200)
    #levels = np.linspace(0.01, 1, 300)
    levels = np.linspace(np.log10(1e-4), np.log10(0.1), 300)
    cs = plt.contourf(a_mesh, b_mesh, np.log10(rcrit_arr), levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')

    # if nfp == 4:
    # fig.colorbar(cs, shrink=0.9)

    # cax = fig.add_subplot(ax)  # Colorbar subplot
    # cbar = fig.colorbar(cs, cax=cax)

    #ax.set_title('nfp = {},      '.format(nfp) + r'$\log_{10} \: r_{\rm crit}$', fontsize = 24)

    ax.set_title(r'$N_{\rm fp}$' + ' = {},      '.format(nfp) + r'$\quad \log_{10} \: r_{\rm crit}$', fontsize = 24)

    #ax.set_title(r'$\log_{10} \: \Delta B_{20}$', fontsize = 18)

    if nfp == 2:

        a_arr = np.linspace(0.0001, 0.6, N)
        line_1 = lambda a: (-0.0612/0.22)*a - 0.0612
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.1171 + 0.0612)/0.6)*a - 0.0612
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.107, 0.269, 0.446, 0.512, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0584, 0.0588, 0.0604, 0.0695, 0.0898, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.45, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.6)
        ax.set_ylim(-0.13, 0.13)


    elif nfp == 3:

        a_arr = np.linspace(0.0001, 0.3, N)
        line_1 = lambda a: (-0.0280/0.108)*a - 0.0280
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0538 + 0.0280)/0.3)*a - 0.0280
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0591, 0.1433, 0.2211, 0.2677, 0.3] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0267, 0.0267, 0.0285, 0.0319, 0.0437, 0.0526] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.2285, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        ax.set_xlim(0.0001, 0.3)
        ax.set_ylim(-0.06, 0.06)


    elif nfp == 4:

        a_arr = np.linspace(0.0001, 0.20, N)
        line_1 = lambda a: (-0.0160/0.064)*a - 0.0160
        ax.plot(a_arr, line_1(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        line_2 = lambda a: ((0.0338 + 0.0160)/0.1891)*a - 0.0160
        ax.plot(a_arr, line_2(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        a_fit_arr = [0.0001, 0.0381, 0.0724, 0.1345, 0.1659, 0.1927] #, 0.1817, 0.2012, 0.2740, 0.2983]
        b_fit_arr = [0.0146, 0.0155, 0.0157, 0.0191, 0.0273, 0.0342] #, 0.01058, 0.01372, 0.00662, -0.04910]

        #ax.plot(a_fit_arr, b_fit_arr, color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)

        # a_fit_arr = [0.0001, 0.446, 0.6] #, 0.1817, 0.2012, 0.2740, 0.2983]
        # b_fit_arr = [0.0584, 0.0695, 0.1171] #, 0.01058, 0.01372, 0.00662, -0.04910]

        curve = np.polyfit(a_fit_arr, b_fit_arr, 3)

        # # use the curve to write a lambda function that plots this curve
        curve = np.poly1d(curve)

        # new array a_arr cut at a certain value
        a_arr = np.linspace(0.0001, 0.1368, 12)

        ax.plot(a_arr, curve(a_arr), color = 'black', linestyle = '-', linewidth = 2, alpha = 0.3)


        ax.set_xlim(0.0001, 0.20)
        ax.set_ylim(-0.0347, 0.0347)


    # add text to figures
    # ax.text(0.7, 0.05, r'$N_{\rm fp}$' + ' = {}'.format(nfp), fontsize = 18, transform=ax.transAxes, color = 'black') 


    # ax.text(0.02, 0.420, r'${\rm QA}($' + '{})'.format(nfp*0), fontsize = 14, transform=ax.transAxes) 

    # ax.text(0.1, 0.85, r'${\rm QH}($' + '{})'.format(nfp*2), fontsize = 14, transform=ax.transAxes) 

    # ax.text(0.02, 0.02, r'${\rm QH}($' + '{})'.format(nfp*2), fontsize = 14, transform=ax.transAxes) 

    # ax.text(0.75, 0.40, r'${\rm QH}($' + '{})'.format(nfp), fontsize = 14, transform=ax.transAxes) 

    # set number of decimal places in axis ticks
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


    if idx == 2:
        ax.set_ylabel(r'$b$', fontsize = 28)
        # put y label on the right
        ax.yaxis.set_label_position("right")

    ax.set_xlabel(r'$a$', fontsize = 28)
    


# cax = fig.add_axes([0.92, 0.14, 0.023, 0.7])  # [left, bottom, width, height]
# cbar = fig.colorbar(cs, cax=cax, orientation='vertical')

cax = fig.add_subplot(grid[1, :])  # Colorbar subplot for plots 1 and 2
cbar = fig.colorbar(cs, cax=cax, orientation='horizontal')
cbar.ax.locator_params(nbins=8)


plt.savefig('configs_to_be_use_rcrit.png', format='png', dpi=300, bbox_inches='tight')
plt.show()



"""

##########################################################################################
# finding a curve to try some things

a_fit_arr = [0.0001, 0.0227, 0.0505, 0.0946, 0.1817, 0.2012, 0.2740, 0.2983]
b_fit_arr = [0.01163, 2e-5, 0.00091, 0.00592, 0.01058, 0.01372, 0.00662, -0.04910]

curve = np.polyfit(a_fit_arr, b_fit_arr, 7)

# use the curve to write a lambda function that plots this curve
curve = np.poly1d(curve)

# print(curve(a_arr))

# get the b_arr closest to the curve
b_fit_curve_arr = curve(a_arr)

# get the all the index of b_mesh that are the closes to b_fit_curve_arr
idx_curve = []
for i in range(len(b_fit_curve_arr)):
    idx_curve.append((i, np.argmin(np.abs(b_arr - b_fit_curve_arr[i]))))
# idx = np.array(idx_cruve)

print(idx_curve)

##########################################################################################

# give the index of the values from delB20_arr with filtering of values less than 0
idx = np.where(delB20_arr < 0.7)
# now make a tuple numpy array of the indices
idx = list(zip(idx[0], idx[1]))

# print(idx)


idx_min = np.unravel_index(np.argmin(delB20_arr, axis=None), delB20_arr.shape)
# print(idx_min)

#plt.figure(figsize = (12/1.5,8/1.5))
fig = plt.figure(figsize = (7/1.5, 6/1.5))

# levels = np.linspace(np.log10(np.min(delB20_arr)), np.log10(np.max(delB20_arr)), 300)
levels = np.linspace(np.log10(np.min(delB20_arr)), 0, 300)
# levels = np.linspace(np.log10(np.min(delB20_arr)), 6, 200)
cs = plt.contourf(a_mesh, b_mesh, np.log10(delB20_arr), levels = 200, cmap = mpl.colormaps['jet'])
fig.colorbar(cs, shrink=0.9)
plt.title('nfp = {},      '.format(nfp) + r'$\log_{10} \: \Delta B_{20}$')

plt.xlabel(r'$a$', fontsize = 28)
plt.ylabel(r'$b$', fontsize = 28)



# # # plot points where delB20 < 0.1
# for i in range(len(idx)):
#     if i == 0:
#         plt.plot(a_mesh[idx[i][0], idx[i][1]], b_mesh[idx[i][0], idx[i][1]], 'k.', markersize = 2, alpha = 0.3, label = r'$\Delta B_{20} < 0.7$')
#     plt.plot(a_mesh[idx[i][0], idx[i][1]], b_mesh[idx[i][0], idx[i][1]], 'k.', markersize = 2, alpha = 0.3)



# plt.plot(a_arr, curve(a_arr), 'k--', label = 'cruve fit')
# plt.plot(a_fit_arr, b_fit_arr, 'k.', label = 'fit points')

# for i in range(len(idx_curve)):
#     if i == 0:
#         plt.plot(a_arr[idx_curve[i][0]], b_arr[idx_curve[i][1]], 'k.', markersize = 3, label = 'curve points')
#     plt.plot(a_arr[idx_curve[i][0]], b_arr[idx_curve[i][1]], 'k.', markersize = 3)

# plt.legend(loc = 'upper right', fontsize = 12)

# idx_min = np.unravel_index(np.argmin(delB20_arr, axis=None), delB20_arr.shape)
# print(idx_min)
# print(np.min(delB20_arr))
# print(a_mesh[idx_min[0], idx_min[1]])
# print(b_mesh[idx_min[0], idx_min[1]])
# # print(np.argmin(delB20_arr))

# plt.show()

# fig = plt.figure(figsize = (7, 6))

# levels = np.linspace(np.min(eta_arr), np.max(eta_arr), 300)
# cs = plt.contourf(a_mesh, b_mesh, eta_arr, levels = levels, cmap = mpl.colormaps['jet'])
# fig.colorbar(cs, shrink=0.9)
# plt.title('nfp = {},  B0 = {},  N = {},  '.format(nfp, B0, N) + r'$\bar{\eta}$')

plt.savefig("THESIS_trial.png", format='png', dpi=300, bbox_inches='tight')

# # plt.show()

##########################################################################################
# plot of r_crit

# give the index of the values from delB20_arr with filtering of values less than 0
idx = np.where(rcrit_arr > 0.05)
# now make a tuple numpy array of the indices
idx = list(zip(idx[0], idx[1]))

# print(idx)

fig = plt.figure(figsize = (7, 6))


# levels = np.linspace(0.01, 0.2, 300)
levels = np.linspace(1e-4, 0.2, 300)
# levels = np.linspace(0.01, 1, 300)
cs = plt.contourf(a_mesh, b_mesh, rcrit_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'max')
fig.colorbar(cs, shrink=0.9)
plt.title('nfp = {},  B0 = {},  N = {},          '.format(nfp, B0, N) + r'$ r_{\rm crit.}$')

# # plot points where rcrit > 0.1
# for i in range(len(idx)):
#     if i == 0:
#         plt.plot(a_mesh[idx[i][0], idx[i][1]], b_mesh[idx[i][0], idx[i][1]], 'k.', markersize = 2, alpha = 0.3, label = r'$r_{\rm crit.} > 0.05$')
#     plt.plot(a_mesh[idx[i][0], idx[i][1]], b_mesh[idx[i][0], idx[i][1]], 'k.', markersize = 2, alpha = 0.3)

# plt.plot(a_arr, curve(a_arr), 'k--', label = 'fit')

plt.xlabel('a')
plt.ylabel('b')
plt.legend(loc = 'upper left', fontsize = 12)

##########################################################################################


# display plot in current screen
plt.show()

"""