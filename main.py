# -*- coding: utf-8 -*-

import numpy as np
import os
from simulator import *
from sinogram import *
from instrument import *


inst = Instrument(612, 3.2)
inst.add_stage_positions([299, 749, 1209, 1666, 2123, 2580, 3038, 3495, 3951, 4409, 4866, 5317])

sim = Simulator()
sim.read_raw_sinogram(os.path.join('test', 'sino_01000.tiff'))
sim.load_instrument(inst)
sim.sample_full_sinogram_tomosaic()

