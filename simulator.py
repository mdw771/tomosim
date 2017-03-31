# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import dxchange
import os, glob, re, warnings
import tomopy
from tomopy import trim_sinogram
from scipy.ndimage import imread

from instrument import *
from sinogram import *
from util import *
from sample import *
from constants import *


class Simulator(object):

    def __init__(self):

        self.sinos_local = []
        self.sinos_tomosaic = []
        self.raw_sino = None
        self.full_recon_local = None
        self.stitched_sino_tomosaic = None
        self.pixel_size = None
        self.sample = None
        self.ds = None
        self.snr_local = None
        self.snr_tomosaic = None

    def read_raw_sinogram(self, fname, type='tiff', center=None, pixel_size=1, **kwargs):
        """
        Read raw sinogram from file.
        :param fname: file name
        :param type: file format
        :param center: rotation center
        :param preprocess: whether or not to preprocess the sinogram to remove singularities
        :param pixel_size: pixel size (um)
        :param kwargs:
        :return:
        """

        if type == 'hdf5':
            slice = kwargs['slice']
            raw_sino = np.squeeze(dxchange.read_aps_32id(fname, sino=(slice, slice+1)))
        else:
            raw_sino = dxchange.read_tiff(fname)
        self.raw_sino = Sinogram(raw_sino, 'raw', coords=center, center=center, normalize_bg=False, minus_log=False)
        self.pixel_size = pixel_size

    def raw_sino_add_noise(self, fraction_mean=0.01):

        self.raw_sino.add_poisson_noise(fraction_mean=fraction_mean)

    def load_instrument(self, instrument):

        assert isinstance(instrument, Instrument)
        self.inst = instrument

    def read_sinos_local(self, read_path=None):

        print('Reading sinograms.')
        flist = glob.glob(os.path.join(read_path, 'sino_loc*'))
        regex = re.compile(r'.+_(\d+)_(\d+).+')
        for fname in flist:
            y, x = map(int, regex.search(fname).group(1, 2))
            data = imread(fname)
            local_sino = Sinogram(data, 'local', coords=(y, x), center=int(self.inst.fov/2))
            self.sinos_local.append(local_sino)

    def sample_full_sinogram_local(self, save_path=None, save_mask=False, direction='clockwise'):
        """
        Extract local tomography sinogram from full sinogram.
        :param save_path:
        :param save_mask:
        :param direction: direction of sample rotation.
               Available options: 'clockwise' or 'anticlockwise'
        :return:
        """

        for center_coords in self.inst.center_positions:

            print('Sampling sinogram for center ({:d}, {:d}).'.format(center_coords[0], center_coords[1]))

            y0, x0 = center_coords
            w = self.raw_sino.shape[1]
            w_2 = int(w / 2)
            fov = self.inst.fov
            fov_2 = int(fov / 2)

            # compute trajectory of center of FOV in sinogram space
            sino, mask = trim_sinogram(self.raw_sino.sinogram[:, np.newaxis, :], self.raw_sino.center, w_2-y0, x0-w_2,
                                       fov)
            sino = np.squeeze(sino)
            dxchange.write_tiff(sino, 'data/temp')

            local_sino = Sinogram(sino, 'local', coords=(y0, x0), center=fov_2, normalize_bg=True, minus_log=True)
            self.sinos_local.append(local_sino)

            if save_path is not None:
                dxchange.write_tiff(local_sino.sinogram, os.path.join(save_path, 'sino_loc_{:d}_{:d}'.format(y0, x0)),
                                    overwrite=True, dtype='float32')
            if save_mask:
                if save_path is None:
                    save_path = 'mask'
                dxchange.write_tiff(mask, os.path.join(save_path, 'mask', 'mask_loc_{:d}_{:d}'.format(y0, x0)),
                                    overwrite=True, dtype='float32')

    def recon_all_local(self, save_path=None, mask_ratio=1):

        for sino in self.sinos_local:
            print('Reconstructing local tomograph at ({:d}, {:d}).'.format(sino.coords[0], sino.coords[1]))
            sino.reconstruct(mask_ratio=mask_ratio)
            if save_path is not None:
                dxchange.write_tiff(sino.recon * sino.recon_mask, os.path.join(save_path, 'recon_loc_{:d}_{:d}'.
                                                                               format(sino.coords[0], sino.coords[1])),
                                    overwrite=True, dtype='float32')

    def stitch_all_recons_local(self, save_path=None, fname='recon_local'):

        self.full_recon_local = np.zeros([self.raw_sino.shape[1], self.raw_sino.shape[1]])
        fov = self.inst.fov
        self.full_recon_local = np.pad(self.full_recon_local, fov, 'constant', constant_values=0)
        for sino in self.sinos_local:
            y, x = sino.coords
            print('Stitching reconstructions at ({:d}, {:d}).'.format(y, x))
            y, x = map(operator.add, (y, x), (fov, fov))
            dy, dx = sino.recon.shape
            dy2, dx2 = map(int, map(operator.div, sino.recon.shape, (2, 2)))
            ystart, xstart = map(operator.sub, (y, x), (dy2, dx2))
            self.full_recon_local[ystart:ystart+dy, xstart:xstart+dx][sino.recon_mask] = sino.recon[sino.recon_mask]
        self.full_recon_local = self.full_recon_local[fov:fov+self.raw_sino.shape[1], fov:fov+self.raw_sino.shape[1]]
        if save_path is not None:
            dxchange.write_tiff(self.full_recon_local, os.path.join(save_path, fname), overwrite=True,
                                dtype='float32')
        return self.full_recon_local

    def sample_full_sinogram_tomosaic(self):

        for center_pos in self.inst.stage_positions:

            sino = np.zeros(self.raw_sino.shape)
            w = sino.shape[1]
            dx2 = int(self.inst.fov / 2)

            endl = center_pos - dx2 if center_pos - dx2 >= 0 else 0
            endr = endl + self.inst.fov if (endl + self.inst.fov <= w) else w

            partial_sino = self.raw_sino.sinogram[:, endl:endr]
            partial_sino = Sinogram(partial_sino, 'tomosaic', coords=center_pos, center=self.raw_sino.center,
                                    normalize_bg=False, minus_log=True)
            self.sinos_tomosaic.append(partial_sino)

    def stitch_all_sinos_tomosaic(self, center=None):

        if center is None:
            center = self.raw_sino.center
        full_sino = np.zeros([1, 1])
        dx2 = int(self.inst.fov / 2)
        for (i, sino) in enumerate(self.sinos_tomosaic):
            print('Stitching tomosaic sinograms ({:d} of {:d} finished).'.format(i+1, len(self.sinos_tomosaic)))
            ledge = sino.coords - dx2 if sino.coords - dx2 >= 0 else 0
            full_sino = arrange_image(full_sino, sino.sinogram, [0, ledge])
        self.stitched_sino_tomosaic = Sinogram(full_sino, 'full', coords=center, center=center)

    def recon_full_tomosaic(self, save_path=None, fname='recon_tomosaic', mask_ratio=1):

        fov = self.stitched_sino_tomosaic.sinogram.shape[1]
        print('Reconstructing full tomosaic sinogram.')
        self.stitched_sino_tomosaic.reconstruct(mask_ratio=mask_ratio)
        self.full_recon_tomosaic = self.stitched_sino_tomosaic.recon
        if save_path is not None:
            dxchange.write_tiff(self.full_recon_tomosaic, os.path.join(save_path, fname), overwrite=True,
                                dtype='float32')
        return self.full_recon_tomosaic

    def estimate_dose(self, energy, sample, flux_rate, exposure, mode='tomosaic'):
        """
        Estimate radiation dose.
        :param flux_rate: photon flux rate (ph/s/mm)
        :param exposure: exposure time (ms)
        :param mode: "tomosaic" or "local"
        :return: radiation energy deposition (J/m^2)
        """
        print('Calculating dose.')
        assert mode in ('tomosaic', 'local') and isinstance(sample, Sample)
        n_proj = self.raw_sino.shape[0]
        n_fov = len(self.inst.stage_positions) if mode == 'tomosaic' else len(self.inst.center_positions)
        fov = self.inst.fov
        fov2 = int(fov / 2)
        w = self.raw_sino.shape[1]
        w2 = w / 2.

        cot = lambda x: 1. / np.tan(x)
        csc = lambda x: 1. / np.sin(x)

        # assume disk sample
        # use intersection length of the central ray as thickness
        e_abs = 0
        n0 = flux_rate * exposure * fov * self.pixel_size * 1e-6
        if mode == 'tomosaic':
            for x0 in self.inst.stage_positions:
                t = 2 * np.sqrt(w2 ** 2 - (w2 - x0) ** 2) * self.pixel_size
                f_abs = 1 - np.exp(-sample.get_attenuation_coeff(energy) * t)
                e_abs += (f_abs * n0 * n_proj) * energy
                # x1 = x0 - fov2 if x0 - fov2 >= 0 else 0
                # x2 = x1 + fov
                # t = 2 * np.sqrt(w2 ** 2 - (w2 - np.arange(x1, x2, dtype='float')) ** 2)
                # t = t * self.pixel_size
                # n_abs = 1 - np.exp(-sample.get_attenuation_coeff(energy) * t)
                # n_abs = np.sum(n_abs) / fov
        else:
            theta_ls = tomopy.angles(n_proj, ang1=0, ang2=180)
            for (y0, x0) in self.inst.center_positions:
                for theta in theta_ls:
                    if theta == 0 or np.abs(theta-np.pi/2) < 1e-6:
                        a = np.abs(w2 - x0)
                    else:
                        a = np.abs((cot(theta) - 1) * w2 + y0 - cot(theta) * x0) / np.abs(csc(theta))
                    if a < w2:
                        t = 2 * np.sqrt(w2 ** 2 - a ** 2) * self.pixel_size
                        f_abs = 1 - np.exp(-sample.get_attenuation_coeff(energy) * t)
                        e_abs += (f_abs * n0) * energy
        e_abs = e_abs * ElectronCharge * 1e3
        return e_abs / (np.pi * (w2 * self.pixel_size * 1e-6) ** 2)

