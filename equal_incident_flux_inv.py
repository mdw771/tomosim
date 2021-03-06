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

    # create reference recon
    if os.path.exists(os.path.join('data', 'ref_recon.tiff')):
        ref_recon = dxchange.read_tiff(os.path.join('data', 'ref_recon.tiff'))
    else:
        sino = dxchange.read_tiff('data/shepp_inv_sino.tiff')
        sino = -np.log(sino)
        sino = sino[:, np.newaxis, :]
        theta = tomopy.angles(sino.shape[0])
        ref_recon = tomopy.recon(sino, theta, center=2048, algorithm='gridrec')
        dxchange.write_tiff(ref_recon, 'data/ref_recon', overwrite=True)
    ref_recon = np.squeeze(ref_recon)

    stage_list = range(256, 4096, 306)
    inst = Instrument(512)
    inst.add_stage_positions(stage_list)

    stage_list = range(256, 4096, 306)
    center_list = [(y, x) for y in stage_list for x in stage_list]
    inst.add_center_positions(center_list)

    n_pos_tomosaic = len(stage_list)
    n_pos_local = len(center_list)

    prj_tomosaic = Project()
    prj_tomosaic.add_simuators(os.path.join('data', 'shepp_inv_sino.tiff'), inst, center=2048, pixel_size=3.2,
                               downsample=(2, 4, 8, 16, 32))
    ds_local = []
    n_proj_full = prj_tomosaic.simulators[0].raw_sino.shape[0]
    for sim in prj_tomosaic.simulators:
        print(sim.raw_sino.shape[0], n_pos_tomosaic, n_pos_local)
        n_proj_local = sim.raw_sino.shape[0] * n_pos_tomosaic / float(n_pos_local)
        ds_local.append(float(n_proj_full) / float(n_proj_local))
    print ds_local

    prj_local = Project()
    prj_local.add_simuators(os.path.join('data', 'shepp_inv_sino.tiff'), inst, center=2048, pixel_size=3.2,
                               downsample=ds_local)

    if True:
        prj_tomosaic.process_all_tomosaic()
        prj_local.process_all_local(mask_ratio=0.85, offset_intensity=True, ref_fname='data/ref_recon.tiff')

    influx = []
    snr_tomosaic = []
    snr_local = []
    sample = Sample('H48.6C32.9N8.9O8.9S0.6', 1.35)
    for sim in prj_tomosaic.simulators[1:]:
        influx.append(sim.estimate_dose_rough(25.7, sample, np.sqrt(1.779e13), 30, mode='tomosaic')[0])
        img = dxchange.read_tiff(os.path.join('data', 'recon_tomosaic_{:s}x.tiff'.format(sim.name_ds)))
        snr_tomosaic.append(snr(img, ref_recon, mask_ratio=0.8))
    for sim in prj_local.simulators[2:]:
        img = dxchange.read_tiff(os.path.join('data', 'recon_local_{:s}x.tiff'.format(sim.name_ds)))
        snr_local.append(snr(img, ref_recon, mask_ratio=0.8))
    plt.figure()
    plt.semilogx(influx, snr_local, label='Local', marker='o')
    plt.semilogx(influx, snr_tomosaic, label='Tomosaic', marker='o')
    plt.legend()
    plt.xlabel('Total influx')
    plt.ylabel('SNR')
    plt.savefig('data/snr_vs_influx.pdf', format='pdf')