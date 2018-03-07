# -*- coding: utf-8 -*-
"""
This script works for shirley sample.
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

import os

np.set_printoptions(threshold='infinite')

data_folder = '/raid/home/mingdu/data/shirley/data'
raw_sino_fname = 'shirley_full_sino.tiff'

pad_length = 0
sino_width = 18710
scanned_sino_width = 18710 + 0 # leave some space at sides to expand FOV
half_sino_width = int(sino_width / 2)

true_center = 9335

n_scan_local_ls = [11]
ovlp_rate_tomosaic = 0.2
mask_ratio_local = 0.7

shift_y = 1700
shift_x = 1700
start_y = 8047
start_x = 3756
tile_y = 2
tile_x = 3
fov = 1920
half_fov = int(fov / 2)

photon_multiplier_ls = [100, 200, 500, 1000, 2000, 5000, 10000]

# create reference recon
if os.path.exists(os.path.join(data_folder, 'ref_recon.tiff')):
    ref_recon = dxchange.read_tiff(os.path.join(data_folder, 'ref_recon.tiff'))
else:
    sino = dxchange.read_tiff(os.path.join(data_folder, raw_sino_fname))
    sino = -np.log(sino)
    sino = sino[:, np.newaxis, :]
    theta = tomopy.angles(sino.shape[0])
    ref_recon = tomopy.recon(sino, theta, center=pad_length+true_center, algorithm='gridrec')
    dxchange.write_tiff(ref_recon, os.path.join(data_folder, 'ref_recon'), overwrite=True)
ref_recon = np.squeeze(ref_recon)

stage_list_y = range(start_y, start_y + tile_y * shift_y + 1, shift_y)
stage_list_x = range(start_x, start_x + tile_x * shift_y + 1, shift_x)
center_list = [(y, x) for y in stage_list_y for x in stage_list_x]

inst = Instrument(fov)
inst.add_center_positions(center_list)

# assuming reading in already minus-logged sinogram
sim = Simulator()
sim.read_raw_sinogram(os.path.join(data_folder, raw_sino_fname),
                      center=pad_length + true_center)

save_path = os.path.join(data_folder, 'local_save')
ref_fname = os.path.join(data_folder, 'ref_recon.tiff')
save_mask = False
allow_read = False
offset_intensity = False

sino_path = os.path.join(save_path, 'sino_loc')
sim.sample_full_sinogram_local(save_path=sino_path)

for ph_mult in photon_multiplier_ls:

    recon_path = os.path.join(save_path, 'recon_loc_phmult_{}'.format(ph_mult))
    sim.recon_all_local(save_path=recon_path, mask_ratio=mask_ratio_local, offset_intensity=offset_intensity,
                        read_internally=False, sino_path=sino_path, poisson_maxcount=ph_mult)

