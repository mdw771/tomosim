# -*- coding: utf-8 -*-

import dxchange
import numpy as np
import tomopy
from util import *


class Sinogram(object):

    def __init__(self, sinogram, type, coords=None, normalize_bg=False, minus_log=False, center=None):

        assert type in ('local', 'tomosaic', 'full', 'raw')

        self.padded = False
        # self.normalized_bg = False
        self.type = type
        self.shape = sinogram.shape # unpadded shape
        if normalize_bg:
            self.scaler = (np.mean(sinogram[:, 0]) + np.mean(sinogram[:, -1])) / 2
            self.padded = True
            # self.normalized_bg = True
            sinogram = tomopy.pad(sinogram[:, np.newaxis, :], 2, npad=int(np.ceil(sinogram.shape[1]*1.5)), mode='edge')
            # sinogram = tomopy.normalize_bg(sinogram)
        if minus_log:
            sinogram = -np.log(sinogram)
        sinogram = np.squeeze(sinogram)
        if self.padded:
            sinogram = lateral_damp(sinogram, length=int(0.3*self.shape[1]))
        self.sinogram = sinogram
        if coords is None:
            self.coords = self.shape[1] / 2
        else:
            self.coords = coords
        if type == 'local':
            if center is None:
                self.center = coords[1]
            else:
                self.center = center
        else:
            if center is None:
                self.center = self.shape[1] / 2
            else:
                self.center = center
        self.recon = None
        self.recon_mask = None

    def reconstruct(self, center=None, mask_ratio=1):

        if center is None:
            center = self.center
        if self.padded:
            ind = int((self.sinogram.shape[1] - self.shape[1]) / 2)
            center += ind
        nang = self.sinogram.shape[0]
        theta = tomopy.angles(nang)

        ###
        dxchange.write_tiff(self.sinogram, 'data/test/test', dtype='float32')
        ###

        data = self.sinogram[:, np.newaxis, :]
        rec = tomopy.recon(data, theta, center=center, algorithm='gridrec')
        rec = np.squeeze(rec)
        if self.padded:
            rec = rec[ind:ind+self.shape[1], ind:ind+self.shape[1]]
        # if self.normalized_bg:
            # rec = rec * self.scaler
        self.recon = rec
        self.recon_mask = tomopy.misc.corr._get_mask(rec.shape[0], rec.shape[1], mask_ratio)

    def add_poisson_noise(self, fraction_mean=0.01):
        """
        Add poisson noise to the sinogram.
        :param fraction_mean: float; poisson expectation as fraction of sinogram mean value
        """
        lam = self.sinogram.mean() * fraction_mean
        noise = np.random.poisson(lam=lam, size=self.sinogram.shape) - lam
        self.sinogram = self.sinogram + noise



