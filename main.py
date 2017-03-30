# -*- coding: utf-8 -*-

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

    stage_list = [306, 738, 1170, 1602, 2034, 2466, 2898, 3330, 3762, 4194, 4626, 5058]
    inst = Instrument(612)
    inst.add_stage_positions(stage_list)

    stage_list = range(306, 5646, 345)
    center_list = [(y, x) for y in stage_list for x in stage_list]
    inst.add_center_positions(center_list)


    prj = Project()
    prj.add_simuators(os.path.join('data', 'sino_raw.tiff'), inst, center=2981, preprocess=True, pixel_size=3.2,
                      downsample=(2, 4, 8))
    prj.process_all_local(save_path='data', save_mask=True, mask_ratio=1)
    prj.process_all_tomosaic(save_path='data')

    sample = Sample('H48.6C32.9N8.9O8.9S0.6', 1.35)

    prj.estimate_dose(25.7, sample, np.sqrt(1.779e13), 30)
    prj.calculate_snr(save_path='data')

    prj.plot_snr_vs_dose()