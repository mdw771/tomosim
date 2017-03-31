# -*- coding: utf-8 -*-
"""
For equal incident photon flux, compare the SNR.
"""

import numpy as np
import glob
import tomopy
import dxchange
import matplotlib.pyplot as plt

from project import *
from simulator import *
from sinogram import *
from instrument import *
from sample import *


if __name__ == '__main__':

    print('This script is for Shepp Logan phantom.\n')

    # shepp = tomopy.shepp2d(4096, dtype='float32')
    # shepp = shepp / shepp.max()
    # dxchange.write_tiff(shepp, 'data/shepp_raw', overwrite=True)
    # theta = tomopy.angles(4096)
    # print('Radon')
    # sino = tomopy.project(shepp.astype('float32'), theta, center=2048, emission=True)
    # print(sino)
    # dxchange.write_tiff(np.squeeze(sino), 'data/shepp_sino', overwrite=True)

    stage_list = range(256, 4096, 306)
    inst = Instrument(612)
    inst.add_stage_positions(stage_list)

    stage_list = range(256, 4096, 306)
    center_list = [(y, x) for y in stage_list for x in stage_list]
    inst.add_center_positions(center_list)

    prj = Project()
    prj.add_simuators(os.path.join('data', 'shepp_sino_trans.tiff'), inst, center=2048, preprocess=True, pixel_size=3.2,
                      downsample=(2, 4, 8))
    prj.process_all_local(save_path='data', save_mask=True, mask_ratio=0.85)
    prj.process_all_tomosaic(save_path='data')

    # sample = Sample('H48.6C32.9N8.9O8.9S0.6', 1.35)

    # prj.estimate_dose(25.7, sample, np.sqrt(1.779e13), 30)
    # prj.calculate_snr(save_path='data')

    # prj.plot_snr_vs_dose()