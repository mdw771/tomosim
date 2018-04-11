# -*- coding: utf-8 -*-
"""
This script works for foam phantom.
"""

import numpy as np
import glob
import dxchange
import matplotlib.pyplot as plt
import scipy.interpolate
import tomopy
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import matplotlib

from project import *
from simulator import *
from sinogram import *
from instrument import *
from sample import *

np.set_printoptions(threshold='infinite')


if __name__ == '__main__':

    pad_length = 1024
    sino_width = 2048
    half_sino_width = int(sino_width / 2)
    # scanned_sino_width = 2048 + 1024

    trunc_ratio_ls = np.arange(0.1, 1.1, 0.1)
    gamma_ps = 0.85
    gamma_os = 0.85
    fprime_ls = (sino_width * trunc_ratio_ls).astype(int)

    mean_count_tomosaic_ls = []
    mean_count_local_ls = []
    dose_integral_tomosaic_ls = []
    dose_integral_local_ls = []

    # create reference recon
    if os.path.exists(os.path.join('data', 'ref_recon.tiff')):
        ref_recon = dxchange.read_tiff(os.path.join('data', 'ref_recon.tiff'))
    else:
        sino = dxchange.read_tiff(os.path.join('data', 'foam_sino_pad.tiff'))
        sino = -np.log(sino)
        sino = sino[:, np.newaxis, :]
        theta = tomopy.angles(sino.shape[0])
        ref_recon = tomopy.recon(sino, theta, center=pad_length+half_sino_width, algorithm='gridrec')
        dxchange.write_tiff(ref_recon, 'data/ref_recon', overwrite=True)
    ref_recon = np.squeeze(ref_recon)

    try:
        mean_count_tomosaic_ls = np.load(os.path.join('data', 'foam_eff_ratio', 'mean_count_tomosaic_ls.npy'))
        mean_count_local_ls = np.load(os.path.join('data', 'foam_eff_ratio', 'mean_count_local_ls.npy'))
        dose_integral_tomosaic_ls = np.load(os.path.join('data', 'foam_eff_ratio', 'dose_integral_tomosaic_ls.npy'))
        dose_integral_local_ls = np.load(os.path.join('data', 'foam_eff_ratio', 'dose_integral_local_ls.npy'))
    except:

        # do things for PS
        for fprime in fprime_ls:

            dirname = 'foam_trunc_{:d}'.format(int(float(fprime / sino_width) * 100))
            f = int(float(fprime) / gamma_ps)
            f2 = int(f / 2)
            print('f\' = {}, f = {}'.format(fprime, f))
            n_scan = get_nscan_ps(f, gamma_ps, sino_width)
            if n_scan == 1:
                stage_list = [pad_length + half_sino_width]
            else:
                stage_begin = pad_length + f2
                stage_list = np.arange(stage_begin, stage_begin + fprime * (n_scan - 1) + 1, fprime, dtype=int)

            inst = Instrument(f)
            inst.add_stage_positions(stage_list)

            prj_tomosaic = Project()
            prj_tomosaic.add_simuators(os.path.join('data', 'foam_sino_pad.tiff'),
                                       inst,
                                       center=pad_length + half_sino_width,
                                       pixel_size=1)

            prj_tomosaic.process_all_tomosaic(save_path=os.path.join('data', 'foam_eff_ratio', dirname),
                                              recon=False)

            mean_count = np.mean(prj_tomosaic.simulators[0].sample_counter_tomosaic)
            mean_count_tomosaic_ls.append(mean_count)

            dose_integral_tomosaic_ls.append(prj_tomosaic.simulators[0].sample_sum_tomosaic)

        # do things for OS
        for fprime in fprime_ls:

            dirname = 'foam_trunc_{:d}'.format(int(float(fprime / sino_width) * 100))
            f = int(float(fprime) / gamma_os)
            f2 = int(f / 2)
            print('f\' = {}'.format(fprime))
            n_scan = get_nscan_os(f, fprime, sino_width)
            if n_scan == 1:
                center_list = [(pad_length + half_sino_width, pad_length + half_sino_width)]
            else:
                stage_begin = pad_length + fprime / np.sqrt(8)
                stage_list = np.arange(stage_begin, stage_begin + fprime / np.sqrt(2) * (n_scan - 1) + 1, fprime / np.sqrt(2), dtype=int)
                center_list = [(y, x) for y in stage_list for x in stage_list]
                center_list_excl = []

            for y, x in center_list:
                if np.linalg.norm(np.array([y, x]) - np.array([pad_length + half_sino_width, pad_length + half_sino_width])) > half_sino_width + fprime / 2:
                    print('({}, {}) skipped because it is too far.'.format(y, x))
                else:
                    center_list_excl.append((y, x))

            inst = Instrument(f)
            inst.add_center_positions(center_list_excl)

            prj_local = Project()
            prj_local.add_simuators(os.path.join('data', 'foam_sino_pad.tiff'),
                                    inst,
                                    center=pad_length + half_sino_width,
                                    pixel_size=1)

            prj_local.process_all_local(mask_ratio=gamma_os,
                                        save_path=os.path.join('data', 'foam_eff_ratio', dirname),
                                        ref_fname=os.path.join('data', 'ref_recon.tiff'),
                                        allow_read=False,
                                        offset_intensity=True,
                                        recon=False)

            mean_count = np.mean(prj_local.simulators[0].sample_counter_local)
            mean_count_local_ls.append(mean_count)

            dose_integral_local_ls.append(prj_local.simulators[0].sample_sum_local)


        mean_count_tomosaic_ls = np.array(mean_count_tomosaic_ls)
        mean_count_local_ls = np.array(mean_count_local_ls)
        dose_integral_tomosaic_ls = np.array(dose_integral_tomosaic_ls)
        dose_integral_local_ls = np.array(dose_integral_local_ls)

        # save
        np.save(os.path.join('data', 'foam_eff_ratio', 'mean_count_tomosaic_ls'), mean_count_tomosaic_ls)
        np.save(os.path.join('data', 'foam_eff_ratio', 'mean_count_local_ls'), mean_count_local_ls)
        np.save(os.path.join('data', 'foam_eff_ratio', 'dose_integral_tomosaic_ls'), dose_integral_tomosaic_ls)
        np.save(os.path.join('data', 'foam_eff_ratio', 'dose_integral_local_ls'), dose_integral_local_ls)

    print(mean_count_tomosaic_ls)
    print(mean_count_local_ls)
    print(dose_integral_tomosaic_ls)
    print(dose_integral_local_ls)

    # x for tomosaic; y for local
    comb_pts = np.array([(x, y) for x in trunc_ratio_ls for y in trunc_ratio_ls])
    area_ratio = np.array([float(x) / y for x in mean_count_tomosaic_ls for y in mean_count_local_ls])
    dose_ratio = np.array([float(x) / y for x in dose_integral_tomosaic_ls for y in dose_integral_local_ls])
    x = comb_pts[:, 0]
    y = comb_pts[:, 1]

    # print eff_ratio.reshape([len(trunc_ratio_tomosaic_ls), len(trunc_ratio_local_ls)])

    t = trunc_ratio_ls
    xx, yy = np.meshgrid(t, t)
    matplotlib.rcParams['pdf.fonttype'] = 'truetype'
    fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 9}
    plt.rc('font', **fontProperties)

    fig = plt.figure(figsize=(8, 7))
    zz = scipy.interpolate.griddata(comb_pts, dose_ratio, (xx, yy), method='linear')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(xx, yy, zz, rstride=5, cstride=5, cmap=cm.jet,
                       linewidth=1, antialiased=True)
    ax.view_init(10, -135)
    ax.set_xlabel('Truncation ratio of PS')
    ax.set_ylabel('Truncation ratio of OS')
    ax.set_zlabel('Dose ratio')
    plt.savefig(os.path.join('data', 'dose_ratio_excl_corners.pdf'), format='pdf')

    fig = plt.figure(figsize=(8, 7))
    zz = scipy.interpolate.griddata(comb_pts, area_ratio, (xx, yy), method='linear')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(xx, yy, zz, rstride=5, cstride=5, cmap=cm.jet,
                           linewidth=1, antialiased=True)
    ax.view_init(10, -135)
    ax.set_xlabel('Truncation ratio of PS')
    ax.set_ylabel('Truncation ratio of OS')
    ax.set_zlabel('Area ratio')
    plt.savefig(os.path.join('data', 'area_ratio_excl_corners.pdf'), format='pdf')

    plt.show()






