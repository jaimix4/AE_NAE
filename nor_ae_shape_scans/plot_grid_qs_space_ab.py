import sys
import numpy as np
import matplotlib.pyplot as plt
from qsc import Qsc
sys.path.append('/Users/jaimecaballero/Desktop/TUe_thesis/code/AEpy-main/AE_NAE_py/')
from ae_func_pyqsc import ae_nae
from ae_nor_func_pyqsc import ae_nor_nae
from opt_eta_star import set_eta_star
from scipy.optimize import minimize
import matplotlib as mpl
import time
from matplotlib.widgets import Slider, Button

# mpl.rcParams['text.usetex'] = True

# from Rodriguez paper

# omt_omn_ratios = [[100, 0], [0, 100], [10, 0], [0, 10], [1, 0], [0, 1], [1, 1], [10, 10], [0.1, 0], [0, 0.1], [0.1, 0.1], [100, 100]]

#for omt_omn_ratio in omt_omn_ratios:

# omt = omt_omn_ratio[0]

# omn = omt_omn_ratio[1]


# a goes from -0.3 to 0.3
a = 0.045

# b goes from -0.06 to 0.06
b = 0.000

rc = np.array([1, a, b])

zs = np.array([0, a, b])

nfp = 3

eta = -0.9

B0 = 1

# minimum of ae_total_arr: 7.90990498818863e-05
# a for minimum of ae_total_arr: 0.105065
# b for minimum of ae_total_arr: -0.0552
# eta for minimum of ae_total_arr: -0.030262679452118205

omn = 3

omt = 1

lam_res = 100000

Delta_r = 1
a_r = 1

r = 1e-5

N = 500
N_less_dense = 200

N_skip = int(N/N_less_dense)

a_arr = np.linspace(0.0001, 0.3, N)
b_arr = np.linspace(-0.06, 0.06, N)

# mesh grid a_arr and b_arr
a_mesh, b_mesh = np.meshgrid(a_arr, b_arr)

shape_mesh = a_mesh.shape

ae_nor_total_arr = np.full(shape_mesh, np.nan)
ae_total_arr = np.full(shape_mesh, np.nan)
eta_arr = np.full(shape_mesh, np.nan)
alpha_tilde_arr = np.full(shape_mesh, np.nan)
iota_arr = np.full(shape_mesh, np.nan)


eta_start = [-0.7]


a_0 = 0.0

a_1 = 0.06

a_2 = 0.15

# a_3 = 0.17

b_0 = -0.06

# for region 1 
b_1_1 = lambda a : -0.035 + ((b_0 - (-0.035))/(0.09 - a_0)) * a

#a_2 = 0.15
b_2_1 = lambda a : -0.017 + ((0.02 - (-0.017))/(a_2 - a_0)) * a

b_3_1 = lambda a: 0.02

b_4_1 = lambda a: 0.035


# for region 2 

b_1_2 = lambda a : -0.035 + ((b_0 - (-0.035))/(0.09 - a_0)) * a

b_2_2 = lambda a : -0.017 + ((b_0 - (-0.035))/(0.09 - a_0)) * a

b_3_2 = lambda a : -0.037 + ((0.038 - (-0.037))/(0.3 - a_0)) * a

b_4_2 = lambda a : -0.017 + ((0.02 - (-0.017))/(a_2 - a_0)) * a

b_5_2 = lambda a: 0.02

b_6_2 = lambda a: 0.035

a_1 = (-0.017 - (-0.037))/((((0.038 - (-0.037))/(0.3 - a_0))) - (((b_0 - (-0.035))/(0.09 - a_0))))

# for region 3 

b_1_3 = lambda a : -0.037 + ((0.038 - (-0.037))/(0.3 - a_0)) * a

slope = (0.06-0.035)/(0.30-0.15)
A_inters = 0.035 - slope*0.15

b_2_3 = lambda a : A_inters + slope * a


def check_region(a, b):

    if a <= a_1:

        if (b >= b_1_1(a) and b <= b_2_1(a)) or (b >= b_3_1(a) and b <= b_4_1(a)):

            return True

    elif a > a_1 and a <= a_2:

        if (b >= b_1_2(a) and b <= b_2_2(a)) or (b >= b_3_2(a) and b <= b_4_2(a)) or (b >= b_5_2(a) and b <= b_6_2(a)):

            return True


    elif a > a_2:

        if b >= b_1_3(a) and b <= b_2_3(a):

            return True

    return False



# actual plot 

fig = plt.figure(figsize = (9, 8.5))

plt.tight_layout()

#grid = plt.GridSpec(11, 11, wspace =0.2, hspace = 0.2) #, height_ratios = [11, 1])
grid = plt.GridSpec(5, 5, wspace = 0.2, hspace = 0.2) #, height_ratios = [11, 1])
#axs_omt_omn
# omt_omn_ratios = [[100, 0], [0, 100], [10, 0], [0, 10], [1, 0], [0, 1], [1, 1], [10, 10], \
#     [0.1, 0], [0, 0.1], [0.1, 0.1], [100, 100]]

omt_omn_search = [[0, 0.1], [0, 1], [0, 10], [0, 100], \
                  [0.1, 0], [1, 0], [10, 0], [100, 0], \
                  [0.1, 0.1], [1, 1], [100, 100], \
                  [1, 10], [4, 6], [8, 2]]
                  #[1, 3]]

omt_omn_axis_labels = [[" ", "0.1"], [" ", 1], [" ", "10"], [" ", "100"], \
                  ["0.1", " "], ["1", " "], ["10", " "], ["100", " "], \
                  [" ", " "], [" ", " "], [" ", " "], \
                  [" ", " "], ["4                      ", "6            "], ["8     ", "                 2"]]

axs = []

# omn, omt  0
axs_0_01 = plt.subplot(grid[-2, 0])
axs.append(axs_0_01)

axs_0_1 = plt.subplot(grid[-3, 0])
axs.append(axs_0_1)

# axs_0_3 = plt.subplot(grid[-4, 0])
# axs.append(axs_0_3)

axs_0_10 = plt.subplot(grid[-4, 0])
axs.append(axs_0_10)

axs_0_100 = plt.subplot(grid[-5, 0])
axs.append(axs_0_100)


# omt, omn = 0

axs_01_0 = plt.subplot(grid[-1, 1])
axs.append(axs_01_0)

axs_1_0 = plt.subplot(grid[-1, 2])
axs.append(axs_1_0)

# axs_3_0 = plt.subplot(grid[-1, 3])
# axs.append(axs_3_0)

axs_10_0 = plt.subplot(grid[-1, 3])
axs.append(axs_10_0)

axs_100_0 = plt.subplot(grid[-1, 4])
axs.append(axs_100_0)

# diognal omn = omt

axs_01_01 = plt.subplot(grid[-2, 1])
axs.append(axs_01_01)

axs_1_1 = plt.subplot(grid[-3, 2])
axs.append(axs_1_1)

# axs_10_10 = plt.subplot(grid[-4, 3])
# axs.append(axs_10_10)

axs_100_100 = plt.subplot(grid[-5, 4])
axs.append(axs_100_100)


# three plots

axs_1_10 = plt.subplot(grid[-4, 2])
axs.append(axs_1_10)

axs_4_6 = plt.subplot(grid[-4, 3])
axs.append(axs_4_6)

axs_8_2 = plt.subplot(grid[-3, 3])
axs.append(axs_8_2)

# 0 0  

axs_0_0 = plt.subplot(grid[-1, 0])
# axs.append(axs_0_0)


# for color bar

axs_everything = plt.subplot(grid[1:-1, -1])
axs_everything.spines['bottom'].set_color('white')
axs_everything.spines['top'].set_color('white')
axs_everything.spines['left'].set_color('white')
axs_everything.spines['right'].set_color('white')
axs_everything.tick_params(labelleft=False, labelbottom=False)
axs_everything.tick_params(axis='x', colors='white')
axs_everything.tick_params(axis='y', colors='white')
pos = axs_everything.get_position()
new_pos = [pos.x0-2, pos.y0, pos.width, pos.height]
axs_everything.set_position(new_pos)


plt.suptitle('nfp = {}, B0 = {} '.format(nfp, B0) + ', ' + r'$r = $' + '{}, '.format(r) + r'$\bar{\eta} = \bar{\eta}_{*}$' + \
    '(max $\iota_N$)', size=16)

for idx_plot, ax in enumerate(axs):

    # nor AE
    omt, omn = omt_omn_search[idx_plot][0], omt_omn_search[idx_plot][1]

    try:
        ae_nor_total_arr = np.log10(np.load('data_plot/ae_nor_total_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.\
        format(omn, omt, N, N_less_dense, nfp)))
    except:
        print("files not found for omn = {}, omt = {}".format(omn, omt))
        ae_nor_total_arr = np.full(shape_mesh, np.nan)
   
    levels = np.linspace(-8, -1.5, 500)
    cs = ax.contourf(a_mesh, b_mesh, ae_nor_total_arr, levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
    #fig.colorbar(cs, ax=ax, shrink=0.9)
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_ylabel(omt_omn_axis_labels[idx_plot][1])#, size=14)
    ax.set_xlabel(omt_omn_axis_labels[idx_plot][0])#, size=14)
    ax.set_xticks([0.15])
    ax.set_yticks([0.0])

        

# x y labels 
# axs_0_100.set_title('omn')
# axs_100_0.set_ylabel('omt')
# axs_100_0.yaxis.set_label_position("right")

# 0 plot
axs_0_0.set_ylabel('omn' +  r'$\rightarrow$' + '  0        ')
axs_0_0.set_xlabel('omt' +  r'$\rightarrow$' + '  0        ')
axs_0_0.plot(a_arr, b_arr, color='white')
# axs_0_0.spines['bottom'].set_color('white')
axs_0_0.spines['top'].set_color('white')
# axs_0_0.spines['left'].set_color('white')
axs_0_0.spines['right'].set_color('white')
axs_0_0.tick_params(labelleft=False, labelbottom=False)
# axs_0_0.tick_params(axis='x', colors='white')
# axs_0_0.tick_params(axis='y', colors='white')
axs_0_0.set_xticks([0.15])
axs_0_0.set_yticks([0.0])
# color bar


# axs_everything.spines['bottom'].set_color('white')
# axs_everything.spines['top'].set_color('white')
# axs_everything.spines['left'].set_color('white')
# axs_everything.spines['right'].set_color('white')
# axs_everything.tick_params(labelleft=False, labelbottom=False)
# axs_everything.tick_params(axis='x', colors='white')
# axs_everything.tick_params(axis='y', colors='white')
# pos = axs_everything.get_position()
# new_pos = [pos.x0+1, pos.y0, pos.width, pos.height]
# axs_everything.set_position(new_pos)
fig.colorbar(cs, ax=axs_everything, shrink=0.9)

# adjusting position of special plots 

# pos = axs_4_6.get_position()
# new_pos = [pos.x0 - 0.03, pos.y0 - 0.01, pos.width, pos.height]
# axs_4_6.set_position(new_pos)

# pos = axs_8_2.get_position()
# new_pos = [pos.x0 - 0.01, pos.y0 + 0.02, pos.width, pos.height]
# axs_8_2.set_position(new_pos)

axs_4_6.set_xticks([0.02])
axs_4_6.set_yticks([-0.03])
axs_4_6.yaxis.tick_right()
axs_4_6.xaxis.tick_top()
axs_4_6.yaxis.set_ticks_position('both')
axs_4_6.xaxis.set_ticks_position('both')
axs_4_6.yaxis.set_label_position('right')
axs_4_6.xaxis.set_label_position('top')
# axs_4_6.tick_params(labelleft=True, labelbottom=True)

axs_8_2.set_xticks([0.12])
axs_8_2.set_yticks([0.0420])
axs_8_2.yaxis.tick_right()
axs_8_2.yaxis.set_ticks_position('both')
axs_8_2.yaxis.set_label_position('right')
# axs_8_2.tick_params(labelleft=True, labelbottom=True)

plt.savefig('ae_nor_total_arr_nfp_{}_B0_{}_r_{}_etastar_max_iotaN_meeting.png'.format(nfp, B0, r), dpi=300)
plt.show()


# from scamp import *

# s = Session()
# clar = s.new_part("clarke")