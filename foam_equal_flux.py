# -*- coding: utf-8 -*-
"""
This script works for foam phantom.
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
    n_scan = 8

    ovlp_rate_tomosaic = 0.2
    mask_ratio_local = 0.9

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

    fov = sino_width if n_scan == 1 else int(sino_width / ((1 - ovlp_rate_tomosaic) * (n_scan - 1) + 1))
    if fov % 2 == 1:
        fov += 1
    half_fov = int(fov / 2)

    stage_list = np.linspace(half_fov+pad_length, sino_width+pad_length-half_fov, n_scan)
    stage_list = stage_list.astype('int')

    inst = Instrument(fov)
    inst.add_stage_positions(stage_list)
    center_list = [(y, x) for y in stage_list for x in stage_list]
    inst.add_center_positions(center_list)

    n_pos_tomosaic = len(stage_list)
    n_pos_local = len(center_list)

    prj_tomosaic = Project()
    prj_tomosaic.add_simuators(os.path.join('data', 'foam_sino_pad.tiff'),
                               inst,
                               center=pad_length+half_sino_width,
                               pixel_size=1,
                               downsample=(2, 4, 8, 16, 32))
    ds_local = []
    n_proj_full = prj_tomosaic.simulators[0].raw_sino.shape[0]
    for sim in prj_tomosaic.simulators:
        n_proj_local = sim.raw_sino.shape[0] * n_pos_tomosaic / float(n_pos_local)
        ds_local.append(float(n_proj_full) / float(n_proj_local))

    prj_local = Project()
    prj_local.add_simuators(os.path.join('data', 'foam_sino_pad.tiff'),
                            inst,
                            center=pad_length+half_sino_width,
                            pixel_size=1,
                            downsample=ds_local)

    prj_tomosaic.process_all_tomosaic(save_path=os.path.join('data', 'foam_eq_flux'))
    prj_local.process_all_local(mask_ratio=mask_ratio_local,
                                offset_intensity=True,
                                ref_fname='data/ref_recon.tiff',
                                save_mask=True,
                                save_path=os.path.join('data', 'foam_eq_flux'))

    # compute influx and plot
    try:
        influx = np.load(os.path.join('data', 'foam_eq_flux', 'influx.npy'))
        snr_tomosaic = np.load(os.path.join('data', 'foam_eq_flux', 'ei_fidelity_tomosaic.npy'))
        snr_local = np.load(os.path.join('data', 'foam_eq_flux', 'ei_fidelity_local.npy'))
    except:
        influx = []
        snr_tomosaic = []
        snr_local = []
        sample = Sample('H48.6C32.9N8.9O8.9S0.6', 1.35)
        for sim in prj_tomosaic.simulators[1:]:
            influx.append(sim.estimate_dose_rough(25.7, sample, np.sqrt(1.78e13), 30, mode='tomosaic')[0])
            img = dxchange.read_tiff(os.path.join('data', 'foam_eq_flux',
                                                  'recon_tomosaic_{:s}x.tiff'.format(sim.name_ds)))
            snr_tomosaic.append(snr(img, ref_recon, mask_ratio=0.4))
        for sim in prj_local.simulators[2:]:
            img = dxchange.read_tiff(os.path.join('data', 'foam_eq_flux', 'recon_local_{:s}x.tiff'.format(sim.name_ds)))
            snr_local.append(snr(img, ref_recon, mask_ratio=0.4))
        np.save(os.path.join('data', 'foam_eq_flux', 'influx'), influx)
        np.save(os.path.join('data', 'foam_eq_flux', 'ei_fidelity_tomosaic'), snr_tomosaic)
        np.save(os.path.join('data', 'foam_eq_flux', 'ei_fidelity_local'), snr_local)

    plt.figure()
    plt.semilogx(influx, snr_local, label='Local', marker='o')
    plt.semilogx(influx, snr_tomosaic, label='Tomosaic', marker='o')
    plt.legend()
    plt.xlabel('Total influx')
    plt.ylabel('Reconstruction fidelity (dB)')
    plt.savefig('data/snr_vs_influx.pdf', format='pdf')
















