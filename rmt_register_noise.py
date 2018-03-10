# -*- coding: utf-8 -*-
"""
This script works for shirley sample.
"""

import numpy as np
import glob
import dxchange
import os
import matplotlib.pyplot as plt
import scipy.interpolate
import tomopy
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import tomosaic

from project import *
from simulator import *
from sinogram import *
from instrument import *
from sample import *


np.set_printoptions(threshold='infinite')

data_folder = '/raid/home/mingdu/data/shirley/local_tomo'
raw_sino_fname = 'shirley_full_sino_no_log.tiff'

pad_length = 0
sino_width = 18710
scanned_sino_width = 18710 + 0 # leave some space at sides to expand FOV
half_sino_width = int(sino_width / 2)

true_center = 9335

ovlp_rate_tomosaic = 0.2
mask_ratio_local = 0.99

shift_y = 1400
shift_x = 1400
start_y = 8047
start_x = 3756
tile_y = 2
tile_x = 3
fov = 1920
half_fov = int(fov / 2)

photon_multiplier_ls = [100, 200, 500, 1000, 2000, 5000, 10000]
# photon_multiplier_ls = [10000, 1000]

# create reference recon
if os.path.exists(os.path.join(data_folder, 'ref_recon.tiff')):
    ref_recon = dxchange.read_tiff(os.path.join(data_folder, 'ref_recon.tiff'))
else:
    sino = dxchange.read_tiff(os.path.join(data_folder, raw_sino_fname))
    sino = -np.log(sino)
    sino = sino[:, np.newaxis, :]
    theta = tomopy.angles(sino.shape[0])
    ref_recon = tomopy.recon(sino, theta, center=pad_length+true_center, algorithm='gridrec', filter_name='parzen')
    dxchange.write_tiff(ref_recon, os.path.join(data_folder, 'ref_recon'), overwrite=True)
ref_recon = np.squeeze(ref_recon)

stage_list_y = range(start_y, start_y + (tile_y - 1) * shift_y + 1, shift_y)
stage_list_x = range(start_x, start_x + (tile_x - 1) * shift_y + 1, shift_x)
center_list = [(y, x) for y in stage_list_y for x in stage_list_x]

inst = Instrument(fov)
inst.add_center_positions(center_list)

# assuming reading in already minus-logged sinogram
sim = Simulator()
sim.read_raw_sinogram(os.path.join(data_folder, raw_sino_fname),
                      center=pad_length + true_center)
sim.load_instrument(inst)

save_path = os.path.join(data_folder, 'local_save')
ref_fname = os.path.join(data_folder, 'ref_recon.tiff')
save_mask = False
allow_read = False
offset_intensity = False

sino_path = os.path.join(save_path, 'sino_loc')
sim.sample_full_sinogram_local(save_path=sino_path)

mean_error_ls = []

for ph_mult in photon_multiplier_ls:

    print('Photon multiplier = {}'.format(ph_mult))

    abs_error_ls = []
    recon_path = os.path.join(save_path, 'recon_loc_phmult_{}'.format(ph_mult))
    sim.recon_all_local(save_path=recon_path, mask_ratio=mask_ratio_local, offset_intensity=offset_intensity,
                        read_internally=False, sino_path=sino_path, poisson_maxcount=ph_mult)
    # register
    for iy, y in enumerate(stage_list_y):
        for ix, x in enumerate(stage_list_x):
            this_img = dxchange.read_tiff(
                os.path.join(save_path, 'recon_loc_phmult_{}'.format(ph_mult), 'recon_loc_{}_{}.tiff'.format(y, x)))
            if ix < len(stage_list_x) - 1:
                right_img = dxchange.read_tiff(
                    os.path.join(save_path, 'recon_loc_phmult_{}'.format(ph_mult), 'recon_loc_{}_{}.tiff'.format(y, stage_list_x[ix+1])))
                print('Registering: ({} {}) with ({} {})'.format(y, x, y, stage_list_x[ix+1]))
                this_shift = tomosaic.create_stitch_shift(this_img, right_img, rangeX=(10, fov - 10))
                abs_error_ls.append(np.abs(this_shift[1] - shift_x))
                print(this_shift)
            if iy < len(stage_list_y) - 1:
                bottom_img = dxchange.read_tiff(
                     os.path.join(save_path, 'recon_loc_phmult_{}'.format(ph_mult), 'recon_loc_{}_{}.tiff'.format(stage_list_y[iy+1], x)))
                print('Registering: ({} {}) with ({} {})'.format(y, x, stage_list_y[iy+1], x))
                this_shift = tomosaic.create_stitch_shift(this_img, bottom_img, down=1, rangeY=(10, fov - 10))
                abs_error_ls.append(np.abs(this_shift[0] - shift_y))
                print(this_shift)
    mean_error_ls.append(np.mean(abs_error_ls))
    print(np.mean(abs_error_ls))
    print('---------------------')

np.save(os.path.join(data_folder, 'mean_error_local'), mean_error_ls)
plt.loglog(photon_multiplier_ls, mean_error_ls, '-o')
plt.show()
