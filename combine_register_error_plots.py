import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# data_folder = '/raid/home/mingdu/data/shirley/local_tomo'
data_folder = '/raid/home/mingdu/data/charcoal/local_tomo'

photon_multiplier_ls = [10, 20, 50, 100, 200, 500, 1000]
mean_error_local = np.load(os.path.join(data_folder, 'mean_error_local.npy'))
mean_error_tomosaic = np.load(os.path.join(data_folder, 'mean_error_tomosaic.npy'))

matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)

plt.semilogx(photon_multiplier_ls, mean_error_local, '-o', label='RMT (reconstruction domain)')
plt.semilogx(photon_multiplier_ls, mean_error_tomosaic, '-o', label='PSMT (projection domain)')
plt.legend()
plt.xlabel('Photon count multiplier')
plt.ylabel('Mean registration error')
# plt.yticks(range(4))
plt.axhline(1, color='black', linestyle='--', linewidth=0.5)
plt.savefig(os.path.join(data_folder, 'error_phmult.pdf'), format='pdf')
plt.show()
