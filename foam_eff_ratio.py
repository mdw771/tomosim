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

from project import *
from simulator import *
from sinogram import *
from instrument import *
from sample import *

np.set_printoptions(threshold='infinite')


if __name__ == '__main__':

    pad_length = 1024
    sino_width = 2048
    half_sino_width = 1024
    scanned_sino_width = 2048 + 1024

    n_scan_tomosaic_ls = np.arange(1, 14, dtype='int')
    n_scan_local_ls = np.arange(1, 14, dtype='int')
    ovlp_rate_tomosaic = 0.2
    mask_ratio_local = 0.8

    trunc_ratio_tomosaic_ls = []
    trunc_ratio_local_ls = []
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
        raise Exception
        mean_count_tomosaic_ls = np.load(os.path.join('data', 'foam_eff_ratio', 'mean_count_tomosaic_ls.npy'))
        mean_count_local_ls = np.load(os.path.join('data', 'foam_eff_ratio', 'mean_count_local_ls.npy'))
        trunc_ratio_tomosaic_ls = np.load(os.path.join('data', 'foam_eff_ratio', 'trunc_ratio_tomosaic_ls.npy'))
        trunc_ratio_local_ls = np.load(os.path.join('data', 'foam_eff_ratio', 'trunc_ratio_local_ls.npy'))
        dose_integral_tomosaic_ls = np.load(os.path.join('data', 'foam_eff_ratio', 'dose_integral_tomosaic_ls.npy'))
        dose_integral_local_ls = np.load(os.path.join('data', 'foam_eff_ratio', 'dose_integral_local_ls.npy'))
    except:

        for n_scan in n_scan_tomosaic_ls:

            print('NSCAN (tomosaic): {:d}'.format(n_scan))

            dirname = 'foam_nscan_{:d}'.format(n_scan)
            try:
                os.mkdir(dirname)
            except:
                pass

            fov = get_fov(n_scan, scanned_sino_width, mask_ratio_local)
            half_fov = int(fov / 2)

            trunc = float(fov) / sino_width
            trunc_ratio_tomosaic_ls.append(trunc)

            stage_begin = ((sino_width + pad_length * 2) - scanned_sino_width) / 2
            stage_end = (sino_width + pad_length * 2) - stage_begin
            stage_list = np.linspace(half_fov+stage_begin, stage_end-half_fov, n_scan)
            stage_list = stage_list.astype('int')

            print('Tomosaic FOV: {}; stage list: {}'.format(fov, stage_list))

            inst = Instrument(fov)
            inst.add_stage_positions(stage_list)

            prj_tomosaic = Project()
            prj_tomosaic.add_simuators(os.path.join('data', 'foam_sino_pad.tiff'),
                                       inst,
                                       center=pad_length+half_sino_width,
                                       pixel_size=1)

            prj_tomosaic.process_all_tomosaic(save_path=os.path.join('data', 'foam_eff_ratio', dirname))

            mean_count = np.mean(prj_tomosaic.simulators[0].sample_counter_tomosaic)
            mean_count_tomosaic_ls.append(mean_count)

            dose_integral_tomosaic_ls.append(prj_tomosaic.simulators[0].sample_sum_tomosaic)

        for n_scan in n_scan_local_ls:

            print('NSCAN (local): {:d}'.format(n_scan))

            dirname = 'foam_nscan_{:d}'.format(n_scan)
            try:
                os.mkdir(os.path.join('data', 'foam_eff_ratio', dirname))
            except:
                pass

            fov = get_fov(n_scan, scanned_sino_width, mask_ratio_local)
            half_fov = int(fov / 2)

            trunc = float(fov) / sino_width
            trunc_ratio_local_ls.append(trunc)

            stage_list = np.linspace(half_fov + pad_length, sino_width + pad_length - half_fov, n_scan)
            stage_list = stage_list.astype('int')
            center_list = [(y, x) for y in stage_list for x in stage_list]

            print('Local FOV: {}; stage list: {}'.format(fov, stage_list))

            inst = Instrument(fov)
            inst.add_center_positions(center_list)

            prj_local = Project()
            prj_local.add_simuators(os.path.join('data', 'foam_sino_pad.tiff'),
                                    inst,
                                    center=pad_length + half_sino_width,
                                    pixel_size=1)

            prj_local.process_all_local(mask_ratio=mask_ratio_local,
                                        save_path=os.path.join('data', 'foam_eff_ratio', dirname),
                                        ref_fname=os.path.join('data', 'ref_recon.tiff'),
                                        allow_read=False,
                                        offset_intensity=True)

            mean_count = np.mean(prj_local.simulators[0].sample_counter_local)
            mean_count_local_ls.append(mean_count)

            dose_integral_local_ls.append(prj_local.simulators[0].sample_sum_local)

        mean_count_tomosaic_ls = np.array(mean_count_tomosaic_ls)
        mean_count_local_ls = np.array(mean_count_local_ls)
        trunc_ratio_tomosaic_ls = np.array(trunc_ratio_tomosaic_ls)
        trunc_ratio_local_ls = np.array(trunc_ratio_local_ls)
        dose_integral_tomosaic_ls = np.array(dose_integral_tomosaic_ls)
        dose_integral_local_ls = np.array(dose_integral_local_ls)

        # save
        np.save(os.path.join('data', 'foam_eff_ratio', 'mean_count_tomosaic_ls'), mean_count_tomosaic_ls)
        np.save(os.path.join('data', 'foam_eff_ratio', 'mean_count_local_ls'), mean_count_local_ls)
        np.save(os.path.join('data', 'foam_eff_ratio', 'trunc_ratio_tomosaic_ls'), trunc_ratio_tomosaic_ls)
        np.save(os.path.join('data', 'foam_eff_ratio', 'trunc_ratio_local_ls'), trunc_ratio_local_ls)
        np.save(os.path.join('data', 'foam_eff_ratio', 'dose_integral_tomosaic_ls'), dose_integral_tomosaic_ls)
        np.save(os.path.join('data', 'foam_eff_ratio', 'dose_integral_local_ls'), dose_integral_local_ls)


    print(trunc_ratio_tomosaic_ls)
    print(trunc_ratio_local_ls)
    print(mean_count_tomosaic_ls)
    print(mean_count_local_ls)

    # x for tomosaic; y for local
    comb_pts = np.array([(x, y) for x in trunc_ratio_tomosaic_ls for y in trunc_ratio_local_ls])
    eff_ratio = np.array([float(x) / y for x in mean_count_tomosaic_ls for y in mean_count_local_ls])
    x = comb_pts[:, 0]
    y = comb_pts[:, 1]

    # print eff_ratio.reshape([len(trunc_ratio_tomosaic_ls), len(trunc_ratio_local_ls)])

    t = np.linspace(0.15, 1, 100)
    xx, yy = np.meshgrid(t, t)
    # rbf = Rbf(x, y, eff_ratio, epsilon=2, function='linear')
    # zz = rbf(xx, yy)
    zz = scipy.interpolate.griddata(comb_pts, eff_ratio, (xx, yy), method='linear')
    # zz = griddata(x, y, eff_ratio, xx, yy, interp='linear')
    print zz
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(xx, yy, zz, rstride=5, cstride=5, cmap=cm.jet,
                       linewidth=1, antialiased=True)
    ax.view_init(10, -135)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.pcolor(xx, yy, zz)
    ax.set_xlabel('Truncation ratio of tomosaic method')
    ax.set_ylabel('Truncation ratio of local tomography method')
    ax.set_zlabel('Dose ratio')
    plt.savefig(os.path.join('data', 'eff_ratio.pdf'), format='pdf')
    plt.show()






