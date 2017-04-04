# -*- coding: utf-8 -*-
"""
This script works for Shepp Logan phantom.
"""

import numpy as np
import glob
import dxchange
import matplotlib.pyplot as plt

from project import *
from simulator import *
from sinogram import *
from instrument import *
from sample import *


if __name__ == '__main__':

    stage_list = range(256, 4096, 306)
    inst = Instrument(512)
    inst.add_stage_positions(stage_list)

    stage_list = range(256, 4096, 306)
    center_list = [(y, x) for y in stage_list for x in stage_list]
    inst.add_center_positions(center_list)

    n_pos_tomosaic = len(stage_list)
    n_pos_local = len(center_list)

    prj_tomosaic = Project()
    prj_tomosaic.add_simuators(os.path.join('data', 'shepp_sino_trans.tiff'), inst, center=2048, pixel_size=3.2,
                               downsample=(2, 4, 8))
    ds_local = []
    n_proj_full = prj_tomosaic.simulators[0].raw_sino.shape[0]
    for sim in prj_tomosaic.simulators:
        print(sim.raw_sino.shape[0], n_pos_tomosaic, n_pos_local)
        n_proj_local = sim.raw_sino.shape[0] * n_pos_tomosaic / float(n_pos_local)
        ds_local.append(float(n_proj_full) / float(n_proj_local))
    print ds_local

    prj_local = Project()
    prj_local.add_simuators(os.path.join('data', 'shepp_sino_trans.tiff'), inst, center=2048, pixel_size=3.2,
                               downsample=ds_local)

    prj_tomosaic.process_all_tomosaic()
    prj_local.process_all_local(mask_ratio=0.85)


    # prj.process_all_local(save_path='data', save_mask=True, mask_ratio=0.9)

    #
    # sample = Sample('H48.6C32.9N8.9O8.9S0.6', 1.35)
    #
    # prj.estimate_dose(25.7, sample, np.sqrt(1.779e13), 30)
    # prj.calculate_snr(save_path='data')
    #
    # prj.plot_snr_vs_dose()