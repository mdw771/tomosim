# -*- coding: utf-8 -*-

import dxchange
import numpy as np
import tomopy


class Sinogram(object):

    def __init__(self, sinogram, type, coords=None, center=None):

        assert type in ('local', 'tomosaic', 'full', 'raw')

        self.type = type
        self.sinogram = sinogram
        if coords is None:
            self.coords = sinogram.shape[1] / 2
        else:
            self.coords = coords
        if type == 'local':
            if center is None:
                self.center = coords[1]
            else:
                self.center = center
        else:
            if center is None:
                self.center = sinogram.shape[1] / 2
            else:
                self.center = center
        self.recon = None
        self.recon_mask = None
        self.shape = sinogram.shape

    def reconstruct(self, center=None, add_mask=False):

        if center is None:
            center = self.center
        nang = self.sinogram.shape[0]
        theta = tomopy.angles(nang)
        data = self.sinogram[:, np.newaxis, :]
        rec = tomopy.recon(data, theta, center=center, algorithm='gridrec')
        rec = np.squeeze(rec)
        self.recon = rec

        if add_mask:
            self.recon_mask = tomopy.misc.corr._get_mask(rec.shape[0], rec.shape[1], 1)

    def add_poisson_noise(self, fraction_mean=0.01):
        """
        Add poisson noise to the sinogram.
        :param fraction_mean: float; poisson expectation as fraction of sinogram mean value
        """

        lam = self.sinogram.mean() * fraction_mean
        noise = np.random.poisson(lam=lam, size=self.sinogram.shape) - lam
        self.sinogram = self.sinogram + noise



