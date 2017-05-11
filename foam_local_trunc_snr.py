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

    n_scan_local_ls = np.arange(1, 14, dtype='int')
    ovlp_rate_tomosaic = 0.2
    mask_ratio_local = 0.9

    trunc_ratio_local_ls = []
    fidelity_local_ls = []

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

        trunc_ratio_local_ls = np.load(os.path.join('data', 'trunc_ratio_local_ls.npy'))
        fidelity_local_ls = np.load(os.path.join('data', 'fidelity_local_ls.npy'))

    except:

        trunc_ratio_local_ls = []
        fidelity_local_ls = []

        for n_scan in n_scan_local_ls:

            print('NSCAN (local): {:d}'.format(n_scan))

            fov = sino_width if n_scan == 1 else int(sino_width / ((1 - ovlp_rate_tomosaic) * (n_scan - 1) + 1))
            if fov % 2 == 1:
                fov += 1
            half_fov = int(fov / 2)
            trunc = float(fov) / sino_width
            trunc_ratio_local_ls = np.append(trunc_ratio_local_ls, trunc)

            dirname = 'foam_nscan_{:d}'.format(n_scan)

            recon = np.squeeze(dxchange.read_tiff(os.path.join('data', 'foam_eff_ratio', dirname, 'recon_local_1x.tiff')))
            fid = snr(recon, ref_recon, mask_ratio=0.4)
            fidelity_local_ls = np.append(fidelity_local_ls, fid)

        fidelity_local_ls = np.array(fidelity_local_ls)
        trunc_ratio_local_ls = np.array(trunc_ratio_local_ls)

    # save
    np.save(os.path.join('data', 'trunc_ratio_local_ls'), trunc_ratio_local_ls)
    np.save(os.path.join('data', 'fidelity_local_ls'), fidelity_local_ls)

    fig = plt.figure()
    plt.plot(trunc_ratio_local_ls, fidelity_local_ls, marker='o')
    plt.xlabel('Trunction ratio of local tomography method')
    plt.ylabel('Reconstruction fidelity (dB)')
    plt.savefig(os.path.join('data', 'local_fidelity.pdf'), format='pdf')
    plt.show()


