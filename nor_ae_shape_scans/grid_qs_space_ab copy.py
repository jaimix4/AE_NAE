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
# mpl.rcParams['text.usetex'] = True

# from Rodriguez paper

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

lam_res = 100000

Delta_r = 1
a_r = 1

omt = 0
omn = 1


# omt_omn_ratios = [[100, 0], [0, 100], [10, 0], [0, 10], [1, 0], [0, 1], [1, 1], [10, 10], [0.1, 0], [0, 0.1], [0.1, 0.1], [100, 100]]

omt_omn_ratios = [[1, 10], [4, 6], [8, 2]]

for omt_omn_ratio in omt_omn_ratios:

    omt = omt_omn_ratio[0]

    omn = omt_omn_ratio[1]


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

    # print(eta_arr)

    # print(ae_nor_total_arr)

    # print(ae_total_arr[0,0])
    # print(ae_total_arr[0,0] == np.nan)
    # print(np.isnan(ae_total_arr[0,0]))

    eta_start = [-0.7]

    # simple  minimazation, better for just optimizing eta

    # def ae_total_function_eta(x):

    #     # Calculate the objective function for the ae_nae function
    #     # for a given value of etabar and a set of fixed parameters.
        
    #     return ae_nae(Qsc(rc=rc, zs=zs, nfp=nfp, etabar=x[0], B0=B0), r, lam_res, Delta_psiA, omn, omT, plot = False)[-1]


    # function for checking which region of the grid is computed in order to define density

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



    # read npy file


    do_it = False

    if do_it == False:

        try:

            ae_nor_total_arr = np.load('data_plot/ae_nor_total_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(omn, omt, N, N_less_dense, nfp))
            ae_total_arr = np.load('data_plot/ae_total_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(omn, omt, N, N_less_dense, nfp))
            eta_arr = np.load('data_plot/eta_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(omn, omt, N, N_less_dense, nfp))
            iota_arr = np.load('data_plot/iota_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(omn, omt, N, N_less_dense, nfp))
            alpha_tilde_arr = np.load('data_plot/alpha_tilde_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(omn, omt, N, N_less_dense, nfp))

            ald_computed = True
            print("files found")

        except:

            ald_computed = False
            print("files not found")

    else:

        ald_computed = False

    if ald_computed == False:

        counter = 0
        total_counter = len(a_arr)*len(b_arr)
        start_time = time.time()

        eta_arr = np.load('data_plot/eta_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(1, 0, N, N_less_dense, nfp))
        iota_arr = np.load('data_plot/iota_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(1, 0, N, N_less_dense, nfp))
        alpha_tilde_arr = np.load('data_plot/alpha_tilde_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(1, 0, N, N_less_dense, nfp))


        for idx, a in enumerate(a_arr):
    
            for idx_b, b in enumerate(b_arr):

                check_reg = check_region(a, b)

                if check_reg == True:

                    rc = np.array([1, a, b])

                    zs = np.array([0, a, b])

                    stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=-0.5, B0=B0)

                    #eta_star = set_eta_star(stel)
                    eta_star = eta_arr[idx_b, idx]

                    stel.etabar = eta_star

                    #eta_arr[idx_b, idx] = eta_star

                    ae_total_arr[idx_b, idx], ae_nor_total_arr[idx_b, idx] = ae_nor_nae(stel, r, lam_res, Delta_r, a_r, omn, omt, plot = False)[-2:]

                    #alpha_tilde_arr[idx_b, idx] = stel.iotaN - stel.iota

                    #iota_arr[idx_b, idx] = np.abs(stel.iotaN)

                elif check_reg == False:

                    if np.isnan(ae_nor_total_arr[idx_b, idx]):

                        # print('paris')

                        rc = np.array([1, a, b])

                        zs = np.array([0, a, b])

                        stel = Qsc(rc=rc, zs=zs, nfp=nfp, etabar=-0.5, B0=B0)

                        #eta_star = set_eta_star(stel)
                        eta_star = eta_arr[idx_b, idx]

                        stel.etabar = eta_star

                        last_idx_b = idx_b + N_skip  if ((idx_b + N_skip + 1) < N) else N 
                        last_idx = idx + N_skip  if ((idx + N_skip + 1) < N) else N 

                        #eta_arr[idx_b : last_idx_b, idx : last_idx] = eta_star

                        ae_total_arr[idx_b : last_idx_b, idx : last_idx], ae_nor_total_arr[idx_b : last_idx_b, idx : last_idx] = ae_nor_nae(stel, r, lam_res, Delta_r, a_r, omn, omt, plot = False)[-2:]

                        #alpha_tilde_arr[idx_b : last_idx_b, idx : last_idx] = stel.iotaN - stel.iota

                        #iota_arr[idx_b : last_idx_b, idx : last_idx] = np.abs(stel.iotaN)

                    else:

                        pass

                counter += 1
                
                if counter % 100 == 0:

                    elapsed_time = time.time() - start_time

                    elapsed_time_min = int(elapsed_time/60) 
                    elapsed_time_sec = int(elapsed_time - elapsed_time_min*60)

                    if elapsed_time_sec < 10:
                        elapsed_time_sec = "0" + str(elapsed_time_sec)

                    eta_time = (elapsed_time/60)*(total_counter/counter-1)

                    eta_time_min = int(eta_time)
                    eta_time_sec = int((eta_time - eta_time_min)*60)

                    if eta_time_sec < 10:
                        eta_time_sec = "0" + str(eta_time_sec)

                    print("###############################################################")
                    print("\nConfigurations done: {}/{}. Elapsed time: {}:{}. ETA: {}:{}. ".format(counter, total_counter, elapsed_time_min, elapsed_time_sec,  eta_time_min, eta_time_sec))
                    # print progress bar
                    print("Progress: [", end="")
                    for i in range(0, int(counter/total_counter*50)):
                        print("=", end="")
                    print(">", end="")
                    for i in range(0, 50-int(counter/total_counter*50)):
                        print(" ", end="")
                    print("] \n")
                    print("###############################################################")


    # save the arrays to npy file
    # np.save('data_plot/ae_nor_total_arr_omn_omt_{:.4f}_{}_{}_etastar.npy'.format(om_min, om_max, N), ae_nor_total_arr)
    # np.save('data_plot/etastar_arr_omn_omt_r_{:.4f}_{}_{}_etastar.npy'.format(om_min, om_max, N), eta_dagger_arr)
    np.save('data_plot/ae_nor_total_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(omn, omt, N, N_less_dense, nfp), ae_nor_total_arr)
    np.save('data_plot/ae_total_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(omn, omt, N, N_less_dense, nfp), ae_total_arr)
    np.save('data_plot/eta_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(omn, omt, N, N_less_dense, nfp), eta_arr)
    np.save('data_plot/iota_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(omn, omt, N, N_less_dense, nfp), iota_arr)
    np.save('data_plot/alpha_tilde_arr_omn_{}_omt_{}_N_{}_{}_nfp_{}_etastar.npy'.format(omn, omt, N, N_less_dense, nfp), alpha_tilde_arr)


    # # print the minimum of ae_total_arr
    # print("minimum of ae_total_arr: {}".format(np.min(ae_nor_total_arr)))
    # # print the a and b for the minimum of ae_total_arr
    # print("a for minimum of ae_total_arr: {}".format(a_mesh[np.where(ae_nor_total_arr == np.min(ae_nor_total_arr))][0]))
    # print("b for minimum of ae_total_arr: {}".format(b_mesh[np.where(ae_nor_total_arr == np.min(ae_nor_total_arr))][0]))
    # # print the eta for the minimum of ae_total_arr
    # print("eta for minimum of ae_total_arr: {}".format(eta_arr[np.where(ae_nor_total_arr == np.min(ae_nor_total_arr))][0]))


    fig = plt.figure(figsize = (12, 8))

    plt.tight_layout()

    grid = plt.GridSpec(2, 6, wspace =0.5, hspace = 0.3, height_ratios = [1.5, 1])

    axs0 = plt.subplot(grid[0, :3])

    axs4 = plt.subplot(grid[0, 3:])

    axs2 = plt.subplot(grid[1, :2])

    axs1 = plt.subplot(grid[1, 2:4])

    axs3 = plt.subplot(grid[1, 4:])


    plt.suptitle('omn = {}, omt = {}, nfp = {}, B0 = {} '.format(omn, omt, nfp, B0) + ', ' + r'$r = $' + '{}, '.format(r) + r'$\bar{\eta} = \bar{\eta}_{*}$' + \
        '(max $\iota_N$)', size=16)

    # nor AE
    levels = np.linspace(np.min(np.log10(ae_nor_total_arr)), -1, 1000)
    levels = np.linspace(-6, -2.71828, 1000)
    cs = axs0.contourf(a_mesh, b_mesh, np.log10(ae_nor_total_arr), levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
    fig.colorbar(cs, ax=axs0, shrink=0.9)

    axs0.set_ylabel('b', size=14)
    axs0.set_xlabel('a', size=14)
    axs0.set_title(r'$\log_{10}(\hat{A}/E_t)$', size=16)

    # lines to check things

    a_0 = 0.0

    a_1 = 0.06

    a_2 = 0.15

    a_3 = 0.17

    b_0 = -0.06

    # for region 1 
    b_1_1 = lambda a : -0.035 + ((b_0 - (-0.035))/(0.09 - a_0)) * a

    #a_2 = 0.15
    b_2_1 = lambda a : -0.017 + ((0.02 - (-0.017))/(a_2 - a_0)) * a

    b_3_1 = lambda a: 0.02*np.ones_like(a)

    b_4_1 = lambda a: 0.035*np.ones_like(a)


    # for region 2 

    b_1_2 = lambda a : -0.035 + ((b_0 - (-0.035))/(0.09 - a_0)) * a

    b_2_2 = lambda a : -0.017 + ((b_0 - (-0.035))/(0.09 - a_0)) * a

    b_3_2 = lambda a : -0.037 + ((0.038 - (-0.037))/(0.3 - a_0)) * a

    b_4_2 = lambda a : -0.017 + ((0.02 - (-0.017))/(a_2 - a_0)) * a

    b_5_2 = lambda a: 0.02*np.ones_like(a)

    b_6_2 = lambda a: 0.035*np.ones_like(a)


    a_1 = (-0.017 - (-0.037))/((((0.038 - (-0.037))/(0.3 - a_0))) - (((b_0 - (-0.035))/(0.09 - a_0))))

    # for region 3 

    b_1_3 = lambda a : -0.037 + ((0.038 - (-0.037))/(0.3 - a_0)) * a

    slope = (0.06-0.035)/(0.30-0.15)
    A_inters = 0.035 - slope*0.15

    b_2_3 = lambda a : A_inters + slope * a


    axs0.set_xlim([np.min(a_arr), np.max(a_arr)])
    axs0.set_ylim([np.min(b_arr), np.max(b_arr)])

    # plot v line
    axs0.axvline(x = a_1, color = 'black', linestyle = '--', linewidth = 1)

    a_arr_r1 = np.linspace(a_0, a_1, 100)

    axs0.plot(a_arr_r1, b_1_1(a_arr_r1), color = 'black', linestyle = '--', linewidth = 1)
    axs0.plot(a_arr_r1, b_2_1(a_arr_r1), color = 'black', linestyle = '--', linewidth = 1)
    axs0.plot(a_arr_r1, b_3_1(a_arr_r1), color = 'black', linestyle = '--', linewidth = 1)
    axs0.plot(a_arr_r1, b_4_1(a_arr_r1), color = 'black', linestyle = '--', linewidth = 1)


    axs0.axvline(x = a_2, color = 'black', linestyle = '--', linewidth = 1)

    a_arr_r2 = np.linspace(a_1, a_2, 100)

    axs0.plot(a_arr_r2, b_1_2(a_arr_r2), color = 'black', linestyle = '--', linewidth = 1)
    axs0.plot(a_arr_r2, b_2_2(a_arr_r2), color = 'black', linestyle = '--', linewidth = 1)
    axs0.plot(a_arr_r2, b_3_2(a_arr_r2), color = 'black', linestyle = '--', linewidth = 1)
    axs0.plot(a_arr_r2, b_4_2(a_arr_r2), color = 'black', linestyle = '--', linewidth = 1)
    axs0.plot(a_arr_r2, b_5_2(a_arr_r2), color = 'black', linestyle = '--', linewidth = 1)
    axs0.plot(a_arr_r2, b_6_2(a_arr_r2), color = 'black', linestyle = '--', linewidth = 1)

    a_arr_r3 = np.linspace(a_2, np.max(a_arr), 100)

    axs0.plot(a_arr_r3, b_1_3(a_arr_r3), color = 'black', linestyle = '--', linewidth = 1)
    axs0.plot(a_arr_r3, b_2_3(a_arr_r3), color = 'black', linestyle = '--', linewidth = 1)


    # AE total
    levels = np.linspace(np.min(np.log10(ae_total_arr)), 0, 1000)
    levels = np.linspace(-8, 0, 1000)
    cs = axs4.contourf(a_mesh, b_mesh, np.log10(ae_total_arr), levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
    fig.colorbar(cs, ax=axs4, shrink=0.9)

    # axs4.set_ylabel('b', size=14)
    axs4.set_xlabel('a', size=14)
    axs4.set_title(r'$\log_{10}(\hat{A})$', size=16)
    axs4.tick_params(labelleft=False)

    # eta 
    cs = axs1.contourf(a_mesh, b_mesh, eta_arr, 1000, cmap = mpl.colormaps['jet'])
    #cs = axs1.contourf(a_mesh, b_mesh, a_mesh*b_mesh, 1000, cmap = mpl.colormaps['jet'])
    fig.colorbar(cs, ax=axs1, shrink=0.9)
    axs1.set_title(r'$\bar{\eta_{*}}}$', size=16)
    axs1.set_ylabel('b', size=14)
    axs1.set_xlabel('a', size=14)
    axs1.tick_params(labelleft=False)

    # alpha tilde
    cs = axs2.contourf(a_mesh, b_mesh, alpha_tilde_arr, 6, cmap = mpl.colormaps['jet'])
    fig.colorbar(cs, ax=axs2, shrink=0.9)
    axs2.set_title(r'$\tilde{\alpha}$', size=16)
    axs2.set_ylabel('b', size=14)
    axs2.set_xlabel('a', size=14)

    # iota
    levels = np.linspace(-2, 2, 1000)
    cs = axs3.contourf(a_mesh, b_mesh, np.log10(iota_arr), levels = levels, cmap = mpl.colormaps['jet'], extend = 'both')
    fig.colorbar(cs, ax=axs3, shrink=0.9)
    axs3.set_title(r'$\log_{10}(|\iota_{N}|)$', size=16)
    axs3.set_xlabel('a', size=14)
    axs3.tick_params(labelleft=False)

    # mark the minimum of ae_total_arr with a circle in red
    # axs0.plot(a_mesh[np.where(ae_nor_total_arr == np.min(ae_nor_total_arr))], b_mesh[np.where(ae_nor_total_arr == np.min(ae_nor_total_arr))], 'ro', alpha = 0.4)
    # axs1.plot(a_mesh[np.where(ae_nor_total_arr == np.min(ae_nor_total_arr))], b_mesh[np.where(ae_nor_total_arr == np.min(ae_nor_total_arr))], 'ro', alpha = 0.4)
    # axs2.plot(a_mesh[np.where(ae_nor_total_arr == np.min(ae_nor_total_arr))], b_mesh[np.where(ae_nor_total_arr == np.min(ae_nor_total_arr))], 'ro', alpha = 0.4)
    # axs3.plot(a_mesh[np.where(ae_nor_total_arr == np.min(ae_nor_total_arr))], b_mesh[np.where(ae_nor_total_arr == np.min(ae_nor_total_arr))], 'ro', alpha = 0.4)

    fig.savefig('qs_ab_space_omn_{}_omt_{}_nfp_{}_N_{}_{}_nor_eta_star_grid_2.png'.format(omn, omt, nfp, N, N_less_dense), format='png', dpi=300, bbox_inches='tight')
    #plt.show()
