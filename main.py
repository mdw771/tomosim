# -*- coding: utf-8 -*-

import numpy as np
import glob
import dxchange
from simulator import *
from sinogram import *
from instrument import *


if __name__ == '__main__':

    stage_list = [306, 738, 1170, 1602, 2034, 2466, 2898, 3330, 3762, 4194, 4626, 5058]

    # tomosaic acquisition

    inst = Instrument(612)
    inst.add_stage_positions(stage_list)

    sim = Simulator()
    sim.read_raw_sinogram(os.path.join('test', 'sino_raw.tiff'), center=2981, preprocess=True, pixel_size=3.2)
    sim.load_instrument(inst)
    # sim.sample_full_sinogram_tomosaic()
    # sim.stitch_all_sinos_tomosaic()
    # sim.recon_full_tomosaic(save_path='test')

    # local acquisition

    center_list = [(y, x) for y in stage_list for x in stage_list]
    inst.add_center_positions(center_list)

    sim.load_instrument(inst)
    sim.read_sinos_local('test/sino_loc')
    if len(glob.glob('test/sino_loc/sino_loc*')) == 0:
        sim.sample_full_sinogram_localtomo(save_path='test/sino_loc', save_mask=True)
    else:
        sim.read_sinos_local('test/sino_loc')
    sim.recon_all_local(save_path='test/recon_loc', mask_ratio=1)
    sim.stitch_all_recons_local(save_path='test')

    sim.estimate_dose(25.7, np.sqrt(1.779e13), 30, mode='tomosaic')