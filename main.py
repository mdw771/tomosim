# -*- coding: utf-8 -*-

import numpy as np
import os
import dxchange
from simulator import *
from sinogram import *
from instrument import *


if __name__ == '__main__':

    stage_list = [299, 749, 1209, 1666, 2123, 2580, 3038, 3495, 3951, 4409, 4866, 5317]

    # tomosaic acquisition

    inst = Instrument(612, 3.2)
    inst.add_stage_positions(stage_list)

    sim = Simulator()
    sim.read_raw_sinogram(os.path.join('test', 'sino_01000.tiff'), center=2981)
    sim.load_instrument(inst)
    sim.sample_full_sinogram_tomosaic()
    sim.stitch_all_sinos_tomosaic()
    rec = sim.recon_full_tomosaic()
    dxchange.write_tiff(rec, 'test/recon_tomosaic', dtype='float32')

    # local acquisition

    center_list = [(y, x) for y in stage_list for x in stage_list]
    inst.add_center_positions(center_list)

    sim.load_instrument(inst)
    sim.sample_full_sinogram_localtomo(save_path='test/sino_loc')
    sim.recon_all_local(save_path='test/recon_loc')
    rec = sim.stitch_all_recons_local()
    dxchange.write_tiff(rec, 'test/recon_local', dtype='float32')