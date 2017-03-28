# -*- coding: utf-8 -*-

from __future__ import print_function
import copy
import glob
import os

import numpy as np
import tomopy

from simulator import *


class Project(object):

    def __init__(self):

        self.simulators = []
        self.downsample = [1]
        self.dose_local = None
        self.dose_tomosaic = None

    def add_simuators(self, fname, instrument, type='tiff', center=None, preprocess=True, pixel_size=1, downsample=None,
                      **kwargs):

        sim = Simulator()
        sim.read_raw_sinogram(fname, type=type, center=center, preprocess=preprocess, pixel_size=pixel_size, **kwargs)
        sim.load_instrument(instrument)
        sim.ds = 1
        self.simulators.append(sim)

        if downsample is not None:
            for ds in downsample:
                sim = copy.deepcopy(self.simulators[0])
                temp = tomopy.downsample(sim.raw_sino.sinogram[:, np.newaxis, :], level=int(np.log2(ds)), axis=0)
                sim.raw_sino.sinogram = np.squeeze(temp)
                sim.raw_sino.shape = sim.raw_sino.sinogram.shape
                sim.ds = ds
                self.simulators.append(sim)

    def process_all_local(self, save_path='data', save_mask=False, mask_ratio=1):

        for sim in self.simulators:

            sino_path = os.path.join(save_path, 'sino_loc_{:d}x'.format(sim.ds))
            if len(glob.glob(os.path.join(sino_path, 'sino_loc*'))) == 0:
                sim.sample_full_sinogram_local(save_path=sino_path, save_mask=save_mask)
            else:
                sim.read_sinos_local(sino_path)

            recon_path = os.path.join(save_path, 'recon_loc_{:d}x'.format(sim.ds))
            sim.recon_all_local(save_path=recon_path, mask_ratio=mask_ratio)
            sim.stitch_all_recons_local(save_path=save_path, fname='recon_local_{:d}x'.format(sim.ds))

    def process_all_tomosaic(self, save_path='data', mask_ratio=1):

        for sim in self.simulators:

            sim.sample_full_sinogram_tomosaic()
            sim.stitch_all_sinos_tomosaic()
            sim.recon_full_tomosaic(save_path=save_path, fname='recon_tomosaic_{:d}x'.format(sim.ds),
                                    mask_ratio=mask_ratio)

    def estimate_dose(self, energy, sample, flux_rate, exposure):

        for sim in self.simulators:

            sim.dose_local = sim.estimate_dose(energy, sample, flux_rate, exposure, mode='local')
            sim.dose_tomosaic = sim.estimate_dose(energy, sample, flux_rate, exposure, mode='tomosaic')

    def calculate_snr(self, save_path='data'):

        ref_local = dxchange.read_tiff(os.path.join(save_path, 'recon_local_1x.tiff'))
        ref_tomosaic = dxchange.read_tiff(os.path.join(save_path, 'recon_tomosaic_1x.tiff'))
        for sim in self.simulators:
            if sim.ds not in (1, None):
                recon_local = dxchange.read_tiff(os.path.join(save_path, 'recon_local_{:d}x.tiff'.format(sim.ds)))
                recon_tomosaic = dxchange.read_tiff(os.path.join(save_path, 'recon_tomosaic_{:d}x.tiff'.format(sim.ds)))
                sim.snr_local = snr(recon_local, ref_local)
                sim.snr_tomosaic = snr(recon_tomosaic, ref_tomosaic)