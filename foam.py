# -*- coding: utf-8 -*-
"""
This script works for Shepp Logan phantom.
"""

import numpy as np
import glob
import dxchange
import matplotlib.pyplot as plt
import tomopy

from project import *
from simulator import *
from sinogram import *
from instrument import *
from sample import *


if __name__ == '__main__':

    pad_length = 1024
    sino_width = 2048
    half_sino_width = 1024

    n_scan_tomosaic_ls = np.arange(13, dtype='int')
    n_scan_local_ls = np.arange(13, dtype='int')
    ovlp_rate_tomosaic = 0.2
    mask_ratio_local = 0.9

    trunc_ratio_tomosaic_ls = []
    trunc_ratio_local_ls = []
    mean_count_tomosaic_ls = []
    mean_count_local_ls = []

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

        mean_count_tomosaic_ls = np.load(os.path.join('data', 'mean_count_tomosaic_ls.npy'))
        mean_count_local_ls = np.load(os.path.join('data', 'mean_count_local_ls.npy'))
        trunc_ratio_tomosaic_ls = np.load(os.path.join('data', 'trunc_ratio_tomosaic_ls.npy'))
        trunc_ratio_local_ls = np.load(os.path.join('data', 'trunc_ratio_local_ls.npy'))

    except:

        for n_scan in n_scan_tomosaic_ls:

            fov = sino_width if n_scan == 1 else int(sino_width / ((1 - ovlp_rate_tomosaic) * (n_scan - 1) + 1))
            if fov % 2 == 1:
                fov += 1
            half_fov = int(fov / 2)

            trunc = float(fov) / sino_width
            trunc_ratio_tomosaic_ls.append(trunc)

            stage_list = np.linspace(half_fov+pad_length, sino_width+pad_length-half_fov, n_scan)
            stage_list = stage_list.astype('int')

            inst = Instrument(fov)
            inst.add_stage_positions(stage_list)

            prj_tomosaic = Project()
            prj_tomosaic.add_simuators(os.path.join('data', 'foam_sino_pad.tiff'),
                                       inst,
                                       center=pad_length+half_sino_width,
                                       pixel_size=1)

            prj_tomosaic.process_all_tomosaic()

            mean_count = np.mean(prj_tomosaic.simulators[0].sample_countet_tomosaic)
            mean_count_tomosaic_ls.append(mean_count)

        for n_scan in n_scan_local_ls:

            fov = sino_width if n_scan == 1 else 2 * sino_width / ((np.sqrt(2)*(n_scan-1) + 2) * mask_ratio_local)
            if fov % 2 == 1:
                fov += 1
            half_fov = int(fov / 2)

            trunc = float(fov) / sino_width
            trunc_ratio_tomosaic_ls.append(trunc)

            stage_list = np.linspace(half_fov + pad_length, sino_width + pad_length - half_fov, n_scan)
            center_list = [(y, x) for y in stage_list for x in stage_list]

            inst = Instrument(fov)
            inst.add_center_positions(center_list)

            prj_local = Project()
            prj_local.add_simuators(os.path.join('data', 'foam_sino_pad.tiff'),
                                    inst,
                                    center=pad_length + half_sino_width,
                                    pixel_size=1)

            prj_local.process_all_local(mask_ratio=mask_ratio_local)

            mean_count = np.mean(prj_local.simulators[0].sample_countet_local)
            mean_count_local_ls.append(mean_count)

        # save
        np.save(mean_count_tomosaic_ls, os.path.join('data', 'mean_count_tomosaic_ls'))
        np.save(mean_count_local_ls, os.path.join('data', 'mean_count_local_ls'))
        np.save(trunc_ratio_tomosaic_ls, os.path.join('data', 'trunc_ratio_tomosaic_ls'))
        np.save(trunc_ratio_local_ls, os.path.join('data', 'trunc_ratio_local_ls'))

    # x for tomosaic; y for local
    comb_pts = np.array([(x, y) for x in trunc_ratio_tomosaic_ls for y in trunc_ratio_local_ls])
    eff_ratio = np.array([float(x) / y for x in mean_count_tomosaic_ls for y in mean_count_local_ls])
    x = comb_pts[:, 0]
    y = comb_pts[:, 1]

    fig = plt.figure()
    plt.scatter(x, y, c=eff_ratio)
    plt.savefig(os.path.join('data', 'eff_ratio.pdf'), format='pdf')



