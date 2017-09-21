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

np.set_printoptions(threshold='infinite')


if __name__ == '__main__':

    sim = Simulator()
    sim.read_raw_sinogram('data/foam_sino_halved.tiff', center=1024)
    inst = Instrument(512)

    sino_width = 2048
    half_sino_width = 1024
    center_pos = [(1024, 844), (1024, 1204)]
    stage_pos = [844, 1204]

    inst.add_center_positions(center_pos)
    inst.add_stage_positions(stage_pos)
    sim.load_instrument(inst)

    sim.sample_full_sinogram_local(save_path='temp/masks', save_mask=True)



