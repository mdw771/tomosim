# -*- coding: utf-8 -*-

import operator
import dxchange
import numpy as np
import tomopy
from util import *


class Sinogram(object):

    def __init__(self, sinogram, type, coords=None, normalize_bg=False, minus_log=False, center=None, fin_angle=180):

        assert type in ('local', 'tomosaic', 'full', 'raw')

        self.padded = False
        # self.normalized_bg = False
        self.type = type
        self.shape = sinogram.shape # unpadded shape
        if normalize_bg:
            self.scaler = (np.mean(sinogram[:, 0]) + np.mean(sinogram[:, -1])) / 2
            self.padded = True
            # self.normalized_bg = True
            sinogram = pad_sinogram(sinogram, int(np.ceil(sinogram.shape[1]*2)))
            # sinogram = tomopy.normalize_bg(sinogram)
        if minus_log:
            sinogram = -np.log(sinogram)
        sinogram = np.squeeze(sinogram)
        # if self.padded:
        #     sinogram = lateral_damp(sinogram, length=int(0.3*self.shape[1]))
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
        self.fin_angle = fin_angle

    def reconstruct(self, center=None, mask_ratio=1):

        if center is None:
            center = self.center
        if self.padded:
            ind = int((self.sinogram.shape[1] - self.shape[1]) / 2)
            center += ind
        nang = self.sinogram.shape[0]
        theta = tomopy.angles(nang, ang1=0, ang2=self.fin_angle)
        data = self.sinogram[:, np.newaxis, :]
        rec = tomopy.recon(data, theta, center=center, algorithm='gridrec')
        rec = np.squeeze(rec)
        if self.padded:
            rec = rec[ind:ind+self.shape[1], ind:ind+self.shape[1]]
        # if self.normalized_bg:
            # rec = rec * self.scaler
        self.recon = rec
        self.recon_mask = tomopy.misc.corr._get_mask(rec.shape[0], rec.shape[1], mask_ratio)

    def add_poisson_noise(self, snr=5):
        """
        Add poisson noise to the sinogram.
        :param fraction_mean: float; poisson expectation as fraction of sinogram mean value
        """
        x_ref_norm_sq = np.linalg.norm(self.sinogram) ** 2
        lam = x_ref_norm_sq / pow(10., snr/10.) / self.sinogram.size
        noise = np.random.poisson(lam=lam, size=self.sinogram.shape) - lam
        self.sinogram = self.sinogram + noise

    def correct_abs_intensity(self, ref):

        mask = tomopy.misc.corr._get_mask(self.recon.shape[0], self.recon.shape[1], 0.6)
        local_mean = np.mean(self.recon[mask])
        y0, x0 = self.coords
        yy, xx = map(int, (self.recon.shape[0] / 2, self.recon.shape[1] / 2))
        ref_spot = ref[y0-yy:y0-yy+self.recon.shape[0], x0-xx:x0-xx+self.recon.shape[1]]
        ref_mean = np.mean(ref_spot[mask])
        dxchange.write_tiff(self.recon, 'tmp/loc')
        dxchange.write_tiff(ref_spot, 'tmp/ref')
        self.recon = self.recon + (ref_mean - local_mean)

