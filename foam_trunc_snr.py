# -*- coding: utf-8 -*-
"""
This script works for foam phantom.
Plot variance and fidelity against truncation ratio.
Will read already-generated data. Use foam_eff_ratio if not exists. 
"""

import numpy as np
import glob
import dxchange
import matplotlib.pyplot as plt
import tomopy
import matplotlib

from project import *
from simulator import *
from sinogram import *
from instrument import *
from sample import *


if __name__ == '__main__':

    pad_length = 1024
    sino_width = 2048
    half_sino_width = 1024
    scanned_sino_width = 2048 + 1024

    n_scan_local_ls = np.arange(1, 14, dtype='int')
    n_scan_tomosaic_ls = np.arange(1, 14, dtype='int')
    ovlp_rate_tomosaic = 0.2
    mask_ratio_local = 0.7

    trunc_ratio_local_ls = []
    fidelity_local_ls = []
    variance_local_ls = []
    variance_local_interior_ls = []
    snr_local_ls = []

    trunc_ratio_tomosaic_ls = []
    fidelity_tomosaic_ls = []
    variance_tomosaic_ls = []
    variance_tomosaic_interior_ls = []
    snr_tomosaic_ls = []

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
        raise ValueError
        trunc_ratio_local_ls = np.load(os.path.join('data', 'trunc_ratio_local_ls.npy'))
        fidelity_local_ls = np.load(os.path.join('data', 'fidelity_local_ls.npy'))
        variance_local_ls = np.load(os.path.join('data', 'variance_local_ls.npy'))
        variance_local_interior_ls = np.load(os.path.join('data', 'variance_local_interior_ls.npy'))
        trunc_ratio_tomosaic_ls = np.load(os.path.join('data', 'trunc_ratio_tomosaic_ls.npy'))
        fidelity_tomosaic_ls = np.load(os.path.join('data', 'fidelity_tomosaic_ls.npy'))
        variance_tomosaic_ls = np.load(os.path.join('data', 'variance_tomosaic_ls.npy'))
        variance_tomosaic_interior_ls = np.load(os.path.join('data', 'variance_tomosaic_interior_ls.npy'))
    except:
        for n_scan in n_scan_tomosaic_ls:
            print('NSCAN (tomosaic): {:d}'.format(n_scan))

            fov = get_fov(n_scan, scanned_sino_width, mask_ratio_local)
            half_fov = int(fov / 2)
            trunc = float(fov) / scanned_sino_width
            trunc_ratio_local_ls.append(trunc)

            dirname = 'foam_nscan_{:d}'.format(n_scan)

            recon = np.squeeze(
                dxchange.read_tiff(os.path.join('data', 'foam_eff_ratio', dirname, 'recon_tomosaic_1x.tiff')))
            fid = snr(recon, ref_recon, mask_ratio=0.4)
            varc = variance(recon)
            fidelity_local_ls.append(fid)
            variance_local_ls.append(varc)

            stage_begin = ((sino_width + pad_length * 2) - scanned_sino_width) / 2
            stage_end = (sino_width + pad_length * 2) - stage_begin
            stage_list = np.linspace(half_fov + stage_begin, stage_end - half_fov, n_scan)
            stage_list = stage_list.astype('int')
            center_list = [(y, x) for y in stage_list for x in stage_list]

            temp = []
            for y, x in center_list:
                img = recon[y - half_fov:y - half_fov + fov, x - half_fov:x - half_fov + fov]
                temp.append(variance(img, mask_ratio=0.4))
            variance_local_interior_ls.append(np.mean(temp))

        for n_scan in n_scan_local_ls:
            print('NSCAN (local): {:d}'.format(n_scan))

            fov = get_fov(n_scan, scanned_sino_width, mask_ratio_local)
            half_fov = int(fov / 2)
            trunc = float(fov) / scanned_sino_width
            trunc_ratio_local_ls.append(trunc)

            dirname = 'foam_nscan_{:d}'.format(n_scan)

            recon = np.squeeze(dxchange.read_tiff(os.path.join('data', 'foam_eff_ratio', dirname, 'recon_local_1x.tiff')))
            fid = snr(recon, ref_recon, mask_ratio=0.4)
            varc = variance(recon)
            fidelity_local_ls.append(fid)
            variance_local_ls.append(varc)

            stage_begin = ((sino_width + pad_length * 2) - scanned_sino_width) / 2
            stage_end = (sino_width + pad_length * 2) - stage_begin
            stage_list = np.linspace(half_fov+stage_begin, stage_end-half_fov, n_scan)
            stage_list = stage_list.astype('int')
            center_list = [(y, x) for y in stage_list for x in stage_list]

            temp = []
            for y, x in center_list:
                img = recon[y-half_fov:y-half_fov+fov, x-half_fov:x-half_fov+fov]
                temp.append(variance(img, mask_ratio=0.4))
            variance_local_interior_ls.append(np.mean(temp))

        fidelity_local_ls = np.array(fidelity_local_ls)
        variance_local_ls = np.array(variance_local_ls)
        variance_local_interior_ls = np.array(variance_local_interior_ls)
        trunc_ratio_local_ls = np.array(trunc_ratio_local_ls)

    # save
    np.save(os.path.join('data', 'trunc_ratio_local_ls'), trunc_ratio_local_ls)
    np.save(os.path.join('data', 'fidelity_local_ls'), fidelity_local_ls)
    np.save(os.path.join('data', 'variance_local_ls'), variance_local_ls)
    np.save(os.path.join('data', 'variance_local_interior_ls'), variance_local_interior_ls)
    np.save(os.path.join('data', 'trunc_ratio_tomosaic_ls'), trunc_ratio_tomosaic_ls)
    np.save(os.path.join('data', 'fidelity_tomosaic_ls'), fidelity_tomosaic_ls)
    np.save(os.path.join('data', 'variance_tomosaic_ls'), variance_tomosaic_ls)
    np.save(os.path.join('data', 'variance_tomosaic_interior_ls'), variance_tomosaic_interior_ls)

    print('===========================')
    print('Local:')
    print('TR: ', trunc_ratio_local_ls)
    print('Fidelity: ', fidelity_local_ls)
    print('Variance: ', variance_local_ls)
    print('Interior variance: ', variance_local_interior_ls)
    print('Tomosaic:')
    print('TR: ', trunc_ratio_tomosaic_ls)
    print('Fidelity: ', fidelity_tomosaic_ls)
    print('Variance: ', variance_tomosaic_ls)
    print('Interior variance: ', variance_tomosaic_interior_ls)
    print('===========================')

    matplotlib.rcParams['pdf.fonttype'] = 'truetype'
    fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 9}
    plt.rc('font', **fontProperties)

    fig = plt.figure()
    plt.plot(trunc_ratio_local_ls, fidelity_local_ls, marker='o', label='RMT fidelity')
    plt.plot(trunc_ratio_tomosaic_ls, fidelity_tomosaic_ls, marker='x', label='PSMT fidelity')
    plt.xlabel('Truncation ratio')
    plt.ylabel('Reconstruction fidelity (dB)')
    plt.legend()
    plt.savefig(os.path.join('data', 'fidelity_trunc.pdf'), format='pdf')

    fig2 = plt.figure()
    plt.plot(trunc_ratio_local_ls, variance_local_ls, marker='o', label='RMT global variance')
    plt.plot(trunc_ratio_tomosaic_ls, variance_tomosaic_ls, marker='x', label='PSMT global variance')
    plt.plot(trunc_ratio_local_ls, variance_local_interior_ls, marker='o', label='RMT interior variance')
    plt.plot(trunc_ratio_tomosaic_ls, variance_tomosaic_interior_ls, marker='x', label='PSMT interior variance')
    plt.legend()
    plt.savefig(os.path.join('data', 'variance_trunc.pdf'), format='pdf')
    plt.show()
