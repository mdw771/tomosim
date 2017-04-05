# -*- coding: utf-8 -*-

import dxchange
import numpy as np
import tomopy


class Sinogram(object):

    def __init__(self, sinogram, type, coords=None, normalize_bg=False, minus_log=False, center=None):

        assert type in ('local', 'tomosaic', 'full', 'raw')

        self.padded = False
        self.type = type
        self.shape = sinogram.shape # unpadded shape
        if normalize_bg:
            self.padded = True
            sinogram = tomopy.pad(sinogram[:, np.newaxis, :], 2, npad=int(np.ceil(sinogram.shape[1]*1.5)), mode='edge')
            # sinogram = tomopy.normalize_bg(sinogram)
        if minus_log:
            sinogram = -np.log(sinogram)
        sinogram = np.squeeze(sinogram)
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
        data = self.sinogram[:, np.newaxis, :]
        rec = tomopy.recon(data, theta, center=center, algorithm='gridrec')
        rec = np.squeeze(rec)
        if self.padded:
            rec = rec[ind:ind+self.shape[1], ind:ind+self.shape[1]]
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



