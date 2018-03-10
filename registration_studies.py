import numpy as np
import dxchange
import tomosaic
import os
import matplotlib.pyplot as plt
import matplotlib


# data_folder = '/raid/home/mingdu/data/VS72_Again_180_25kev_lens10x_dfocus12cm_76_y1_x0/localtomo'
# data_folder = '/raid/home/mingdu/data/shirley/local_tomo'
data_folder = '/raid/home/mingdu/data/charcoal/local_tomo'
# data_folder = '/raid/home/mingdu/data/SAMPLE_03/panos'
# full_proj_fname = 'proj_raw_mlog.tiff'
full_proj_fname = '0_norm.tiff'
# full_proj_fname = 'frame0900-2.tiff'
tile_size = (1024, 1024)
half_tile_size = np.floor((np.array(tile_size) / 2)).astype('int')
shift = 850
central_slice = 10068
full_proj = dxchange.read_tiff(os.path.join(data_folder, full_proj_fname))
full_proj = np.squeeze(full_proj)
# pos_ls = range(0, full_proj.shape[-1] - tile_size[0] + 1, shift)
pos_ls = range(0, 12816 - tile_size[0] + 1, shift)
photon_multiplier_ls = [100, 200, 500, 1000, 2000, 5000, 10000]

mean_diff_ls = []

for ph_mult in photon_multiplier_ls:
    print('Multiplier: {}'.format(ph_mult))
    abs_diff_ls = []
    for i, pos in enumerate(pos_ls):
        tile = full_proj[central_slice - half_tile_size[0]:central_slice + half_tile_size[0], pos:pos + tile_size[1]]
        tile = np.exp(-tile) * ph_mult
        tile = np.random.poisson(tile)
        tile = -np.log(tile)
        if i >= 1:
            this_shift = tomosaic.create_stitch_shift(cache, tile, remove_border=0)
            print(this_shift)
            abs_diff_ls.append(np.abs(this_shift[1] - shift))
        dxchange.write_tiff(tile, os.path.join(data_folder, 'proj_tiles', '{}'.format(ph_mult), '{:02d}'.format(i)), dtype='float32', overwrite=True)
        cache = tile
    print(np.mean(abs_diff_ls))
    mean_diff_ls.append(np.mean(abs_diff_ls))
    print('-------------')

matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)

plt.figure(figsize=(8, 5))
plt.loglog(photon_multiplier_ls, mean_diff_ls, '-o')
plt.xlabel('Photon count multiplier')
plt.ylabel('Mean registration error')
plt.savefig(os.path.join(data_folder, 'error_phmult'), format='pdf')
plt.show()
