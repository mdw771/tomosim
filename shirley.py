# -*- coding: utf-8 -*-
"""
This script works for shirley sample.
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

    pad_length = 0
    sino_width = 18710
    scanned_sino_width = 18710 + 0 # leave some space at sides to expand FOV
    half_sino_width = int(sino_width / 2)

    true_center = 9335

    # n_scan_tomosaic_ls = np.arange(1, 14, dtype='int')
    n_scan_tomosaic_ls = []
    # n_scan_local_ls = np.arange(1, 14, dtype='int')
    n_scan_local_ls = [11]
    ovlp_rate_tomosaic = 0.2
    mask_ratio_local = 0.7

    trunc_ratio_tomosaic_ls = []
    trunc_ratio_local_ls = []
    mean_count_tomosaic_ls = []
    mean_count_local_ls = []

    # create reference recon
    if os.path.exists(os.path.join('data', 'ref_recon.tiff')):
        ref_recon = dxchange.read_tiff(os.path.join('data', 'ref_recon.tiff'))
    else:
        sino = dxchange.read_tiff(os.path.join('data', 'shirley_full_sino.tiff'))
        sino = -np.log(sino)
        sino = sino[:, np.newaxis, :]
        theta = tomopy.angles(sino.shape[0])
        ref_recon = tomopy.recon(sino, theta, center=pad_length+true_center, algorithm='gridrec')
        dxchange.write_tiff(ref_recon, 'data/ref_recon', overwrite=True)
    ref_recon = np.squeeze(ref_recon)

    try:
        raise Exception
        mean_count_tomosaic_ls = np.load(os.path.join('data', 'shirley_local', 'mean_count_tomosaic_ls.npy'))
        mean_count_local_ls = np.load(os.path.join('data', 'shirley_local', 'mean_count_local_ls.npy'))
        trunc_ratio_tomosaic_ls = np.load(os.path.join('data', 'shirley_local', 'trunc_ratio_tomosaic_ls.npy'))
        trunc_ratio_local_ls = np.load(os.path.join('data', 'shirley_local', 'trunc_ratio_local_ls.npy'))
    except:

        # for n_scan in n_scan_tomosaic_ls:
        # 
        #     print('NSCAN (tomosaic): {:d}'.format(n_scan))
        # 
        #     dirname = 'shirley_nscan_{:d}'.format(n_scan)
        #     try:
        #         os.mkdir(dirname)
        #     except:
        #         pass
        # 
        #     fov = get_fov(n_scan, scanned_sino_width, mask_ratio_local)
        #     half_fov = int(fov / 2)
        # 
        #     trunc = float(fov) / sino_width
        #     trunc_ratio_tomosaic_ls.append(trunc)
        # 
        #     stage_begin = ((sino_width + pad_length * 2) - scanned_sino_width) / 2
        #     stage_end = (sino_width + pad_length * 2) - stage_begin
        #     stage_list = np.linspace(half_fov+stage_begin, stage_end-half_fov, n_scan)
        #     stage_list = stage_list.astype('int')
        # 
        #     inst = Instrument(fov)
        #     inst.add_stage_positions(stage_list)
        # 
        #     prj_tomosaic = Project()
        #     prj_tomosaic.add_simuators(os.path.join('data', 'shirley_sino_pad.tiff'),
        #                                inst,
        #                                center=pad_length+half_sino_width,
        #                                pixel_size=1)
        # 
        #     prj_tomosaic.process_all_tomosaic(save_path=os.path.join('data', 'shirley_local', dirname))
        # 
        #     mean_count = np.mean(prj_tomosaic.simulators[0].sample_counter_tomosaic)
        #     mean_count_tomosaic_ls.append(mean_count)

        for n_scan in n_scan_local_ls:

            print('NSCAN (local): {:d}'.format(n_scan))

            dirname = 'shirley_nscan_{:d}'.format(n_scan)
            try:
                os.mkdir(os.path.join('data', 'shirley_local', dirname))
            except:
                pass

            fov = get_fov(n_scan, scanned_sino_width, mask_ratio_local)
            print(fov)
            half_fov = int(fov / 2)

            trunc = float(fov) / sino_width
            trunc_ratio_local_ls.append(trunc)

            stage_begin = ((sino_width + pad_length * 2) - scanned_sino_width) / 2
            stage_end = (sino_width + pad_length * 2) - stage_begin
            stage_list = np.linspace(half_fov+stage_begin, stage_end-half_fov, n_scan)
            stage_list = stage_list.astype('int')
            center_list = [(y, x) for y in stage_list for x in stage_list]

            inst = Instrument(fov)
            inst.add_center_positions(center_list)

            prj_local = Project()
            prj_local.add_simuators(os.path.join('data', 'shirley_full_sino.tiff'),
                                    inst,
                                    center=pad_length + true_center,
                                    pixel_size=1)

            save_path = os.path.join('data', 'shirley_local', dirname)
            ref_fname = os.path.join('data', 'ref_recon.tiff')
            save_mask = False
            allow_read = False
            offset_intensity = False
            mask_ratio = mask_ratio_local

            for sim in prj_local.simulators:

                sino_path = os.path.join(save_path, 'sino_loc_{:s}x'.format(sim.name_ds))

                # if len(glob.glob(os.path.join(sino_path, 'sino_loc*'))) == 0:
                #     sim.sample_full_sinogram_local(save_path=sino_path,
                #                                    save_mask=save_mask,
                #                                    fin_angle=180,
                #                                    save_internally=False)
                # else:
                #     if allow_read:
                #         sim.read_sinos_local(sino_path, fin_angle=180)
                #     else:
                #         sim.sample_full_sinogram_local(save_path=sino_path, save_mask=save_mask,
                #                                        fin_angle=180, save_internally=False)

                recon_path = os.path.join(save_path, 'recon_loc_{:s}x'.format(sim.name_ds))
                print('RECON ALL')
                sim.recon_all_local(save_path=recon_path, mask_ratio=mask_ratio, offset_intensity=offset_intensity,
                                    read_internally=False, sino_path=sino_path)
                sim.stitch_all_recons_local(save_path=save_path, fname='recon_local_{:s}x'.format(sim.name_ds))



            mean_count = np.mean(prj_local.simulators[0].sample_counter_local)
            mean_count_local_ls.append(mean_count)

        mean_count_tomosaic_ls = np.array(mean_count_tomosaic_ls)
        mean_count_local_ls = np.array(mean_count_local_ls)
        trunc_ratio_tomosaic_ls = np.array(trunc_ratio_tomosaic_ls)
        trunc_ratio_local_ls = np.array(trunc_ratio_local_ls)

        # save
        np.save(os.path.join('data', 'shirley_local', 'mean_count_tomosaic_ls'), mean_count_tomosaic_ls)
        np.save(os.path.join('data', 'shirley_local', 'mean_count_local_ls'), mean_count_local_ls)
        np.save(os.path.join('data', 'shirley_local', 'trunc_ratio_tomosaic_ls'), trunc_ratio_tomosaic_ls)
        np.save(os.path.join('data', 'shirley_local', 'trunc_ratio_local_ls'), trunc_ratio_local_ls)

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
    plt.xlabel('Truncation ratio of tomosaic method')
    plt.ylabel('Truncation ratio of local tomography method')
    plt.savefig(os.path.join('data', 'eff_ratio.pdf'), format='pdf')
    plt.show()
