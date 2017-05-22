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

np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':

    max_count_ls = [0, 100, 1000, 10000]

    pad_length = 1024
    sino_width = 2048
    half_sino_width = 1024

    n_scan = 8
    ovlp_rate_tomosaic = 0.2
    mask_ratio_local = 0.9

    mean_fidelity_ls = []

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
        max_count_ls = np.load(os.path.join('data', 'foam_noise_contrib', 'maxcount_ls.npy'))
        mean_fidelity_ls = np.load(os.path.join('data', 'foam_noise_contrib', 'mean_fidelity_ls.npy'))
    except:

        for max_count in max_count_ls:

            print('MAX_COUNT (local): {:d}'.format(max_count))

            dirname = 'foam_maxcount_{:d}'.format(max_count)
            if max_count == 0:
                max_count = None
            try:
                os.mkdir(os.path.join('data', 'foam_noise_contrib', dirname))
            except:
                pass

            fov = sino_width if n_scan == 1 else 2 * sino_width / ((np.sqrt(2)*(n_scan-1) + 2) * mask_ratio_local)
            if fov % 2 == 1:
                fov += 1
            half_fov = int(fov / 2)

            trunc = float(fov) / sino_width

            stage_list = np.linspace(half_fov + pad_length, sino_width + pad_length - half_fov, n_scan)
            stage_list = stage_list.astype('int')
            center_list = [(y, x) for y in stage_list for x in stage_list]

            inst = Instrument(fov)
            inst.add_center_positions(center_list)

            prj_local = Project()
            prj_local.add_simuators(os.path.join('data', 'foam_sino_pad.tiff'),
                                    inst,
                                    center=pad_length + half_sino_width,
                                    pixel_size=1,
                                    max_count=max_count)

            prj_local.process_all_local(mask_ratio=mask_ratio_local,
                                        save_path=os.path.join('data', 'foam_noise_contrib', dirname),
                                        ref_fname=os.path.join('data', 'ref_recon.tiff'),
                                        allow_read=False,
                                        offset_intensity=True)

            snr_ls = []
            for y, x in center_list:
                img = dxchange.read_tiff(os.path.join('data',
                                                      'foam_noise_contrib',
                                                      dirname,
                                                      'recon_loc_1x',
                                                      'recon_loc_{:d}_{:d}'.format(y, x)))
                ref = ref_recon[y-half_fov:y-half_fov+fov, x-half_fov:x-half_fov+fov]
                dxchange.write_tiff(img, 'data/foam_noise_contrib/tmp/tmp', dtype='float32')
                dxchange.write_tiff(ref, 'data/foam_noise_contrib/tmp/ref', dtype='float32')
                snr_temp = snr(img, ref, mask_ratio=0.7)
                snr_ls.append(snr_temp)
            snr_mean = np.mean(snr_ls)
            mean_fidelity_ls.append(snr_mean)

            # save
            np.save(os.path.join('data', 'foam_noise_contrib', 'maxcount_ls'), max_count_ls)
            np.save(os.path.join('data', 'foam_noise_contrib', 'mean_fidelity_ls'), mean_fidelity_ls)

    print(max_count_ls)
    print(mean_fidelity_ls)

    fig, ax = plt.subplots()
    extra_roi_fidelity_ls = [mean_fidelity_ls[0]] * len(mean_fidelity_ls)

    ax.barh(max_count_ls, mean_fidelity_ls, max_count_ls, extra_roi_fidelity_ls)

    plt.savefig(os.path.join('data', 'foam_noise_contrib.pdf'), format='pdf')
    plt.show()


