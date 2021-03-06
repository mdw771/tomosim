# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import dxchange
import os, glob, re, warnings
import tomopy
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
        self.name_ds = None
        self.snr_local = None
        self.snr_tomosaic = None
        self.sample_counter_tomosaic = None
        self.sample_counter_local = None
        self.sample_sum_tomosaic = None
        self.sample_sum_local = None

    def read_raw_sinogram(self, fname, type='tiff', center=None, pixel_size=1, fin_angle=180,
                          max_count=None, **kwargs):
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
        raw_sino = np.copy(raw_sino)
        self.raw_sino = Sinogram(raw_sino, 'raw', coords=center, center=center, normalize_bg=False, minus_log=False,
                                 fin_angle=fin_angle, max_count=max_count)
        self.pixel_size = pixel_size

    def load_instrument(self, instrument):

        assert isinstance(instrument, Instrument)
        self.inst = instrument

    def read_sinos_local(self, read_path=None, fin_angle=180):

        print('Reading sinograms.')
        flist = glob.glob(os.path.join(read_path, 'sino_loc*'))
        regex = re.compile(r'.+_(\d+)_(\d+).+')
        for fname in flist:
            y, x = map(int, regex.search(fname).group(1, 2))
            data = imread(fname)
            local_sino = Sinogram(data, 'local', coords=(y, x), center=int(self.inst.fov/2), normalize_bg=True,
                                  minus_log=True, fin_angle=fin_angle)
            self.sinos_local.append(local_sino)

    def sample_full_sinogram_local(self, save_path=None, save_mask=False, direction='clockwise', fin_angle=180,
                                   save_internally=True, verbose=False):
        """
        Extract local tomography sinogram from full sinogram.
        :param save_path:
        :param save_mask:
        :param direction: direction of sample rotation.
               Available options: 'clockwise' or 'anticlockwise'
        :return:
        """
        self.sample_counter_local = np.zeros(self.raw_sino.shape)
        self.sample_sum_local = 0

        for center_coords in self.inst.center_positions:

            if verbose:
                print('Sampling sinogram for center ({:d}, {:d}).'.format(center_coords[0], center_coords[1]))

            y0, x0 = center_coords
            w = self.raw_sino.shape[1]
            w_2 = int(w / 2)
            fov = self.inst.fov
            fov_2 = int(fov / 2)

            # compute trajectory of center of FOV in sinogram space
            sino, mask = trim_sinogram(self.raw_sino.sinogram[:, np.newaxis, :], self.raw_sino.center, w_2-y0, x0-w_2,
                                       fov, fin_angle=fin_angle)
            sino = np.squeeze(sino)
            # dxchange.write_tiff(sino, 'data/temp')

            if save_internally:
                local_sino = Sinogram(np.copy(sino), 'local', coords=(y0, x0), center=fov_2, normalize_bg=True, minus_log=True,
                                      fin_angle=fin_angle)
                self.sinos_local.append(local_sino)

            if save_path is not None:
                dxchange.write_tiff(sino, os.path.join(save_path, 'sino_loc_{:d}_{:d}'.format(y0, x0)),
                                    overwrite=True, dtype='float32')
            if save_mask:
                if save_path is None:
                    save_path = 'mask'
                dxchange.write_tiff(mask, os.path.join(save_path, 'mask', 'mask_loc_{:d}_{:d}'.format(y0, x0)),
                                    overwrite=True, dtype='float32')

            if save_internally:
                self.sample_counter_local[mask] += 1
                self.sample_sum_local += np.sum(-np.log(sino))

    def recon_all_local(self, save_path=None, mask_ratio=1, offset_intensity=False, read_internally=True,
                        sino_path=None, padded_length=0, poisson_maxcount=None, remove_ring=False, **kwargs):

        if read_internally:
            for sino in self.sinos_local:
                print('Reconstructing local tomograph at ({:d}, {:d}).'.format(sino.coords[0], sino.coords[1]))
                sino.reconstruct(mask_ratio=mask_ratio, poisson_maxcount=poisson_maxcount, remove_ring=remove_ring)
                if offset_intensity:
                    fname = kwargs['ref_fname']
                    ref = np.squeeze(dxchange.read_tiff(fname))
                    sino.correct_abs_intensity(ref)
                if save_path is not None:
                    dxchange.write_tiff(sino.recon, os.path.join(save_path, 'recon_loc_{:d}_{:d}'.
                                                                                   format(sino.coords[0], sino.coords[1])),
                                        overwrite=True, dtype='float32')
        else:
            flist = glob.glob(os.path.join(sino_path, '*.tiff'))
            for f in flist:
                sino = dxchange.read_tiff(f)
                coords = re.findall('\d+', f)[-2:]
                coords = map(int, coords)
                print('Reconstructing local tomograph at ({:d}, {:d}).'.format(coords[0], coords[1]))
                sino = Sinogram(sino, 'local', coords=coords, center=int(self.inst.fov / 2), normalize_bg=True,
                                minus_log=True)
                sino.reconstruct(mask_ratio=mask_ratio, poisson_maxcount=poisson_maxcount, remove_ring=remove_ring)
                if offset_intensity:
                    fname = kwargs['ref_fname']
                    ref = np.squeeze(dxchange.read_tiff(fname))
                    sino.correct_abs_intensity(ref)
                if save_path is not None:
                    dxchange.write_tiff(sino.recon, os.path.join(save_path, 'recon_loc_{:d}_{:d}'.
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

    def sample_full_sinogram_tomosaic(self, fin_angle=180):

        self.sample_counter_tomosaic = np.zeros(self.raw_sino.shape)
        self.sample_sum_tomosaic = 0

        for center_pos in self.inst.stage_positions:

            sino = np.zeros(self.raw_sino.shape)
            w = sino.shape[1]
            dx2 = int(self.inst.fov / 2)

            endl = center_pos - dx2 if center_pos - dx2 >= 0 else 0
            endr = endl + self.inst.fov if (endl + self.inst.fov <= w) else w

            partial_sino = self.raw_sino.sinogram[:, endl:endr]
            partial_sino = Sinogram(partial_sino, 'tomosaic', coords=center_pos, center=self.raw_sino.center,
                                    normalize_bg=False, minus_log=True, fin_angle=fin_angle)
            self.sinos_tomosaic.append(partial_sino)

            self.sample_counter_tomosaic[:, endl:endr] += 1
            self.sample_sum_tomosaic += np.sum(partial_sino.sinogram)

    def stitch_all_sinos_tomosaic(self, center=None):

        # if center is None:
        #     center = self.raw_sino.center
        # full_sino = np.zeros([1, 1])
        # dx2 = int(self.inst.fov / 2)
        # for (i, sino) in enumerate(self.sinos_tomosaic):
        #     print('Stitching tomosaic sinograms ({:d} of {:d} finished).'.format(i+1, len(self.sinos_tomosaic)))
        #     ledge = sino.coords - dx2 if sino.coords - dx2 >= 0 else 0
        #     full_sino = arrange_image(full_sino, sino.sinogram, [0, ledge])
        # self.stitched_sino_tomosaic = Sinogram(full_sino, 'full', coords=center, center=center, fin_angle=sino.fin_angle)
        if center is None:
            center = self.raw_sino.center
        full_sino = np.zeros(self.raw_sino.shape)
        dx2 = int(self.inst.fov / 2)
        for (i, sino) in enumerate(self.sinos_tomosaic):
            print('Stitching tomosaic sinograms ({:d} of {:d} finished).'.format(i + 1, len(self.sinos_tomosaic)))
            full_sino[:, sino.coords-dx2:sino.coords-dx2+self.inst.fov] = sino.sinogram
        self.stitched_sino_tomosaic = Sinogram(full_sino, 'full', coords=center, center=center,
                                               fin_angle=sino.fin_angle)


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

    def estimate_dose_rough(self, energy, sample, flux_rate, exposure, mode='tomosaic'):
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
        w = self.raw_sino.shape[1]
        w2 = w / 2.

        # total incident flux
        influx = flux_rate * exposure * fov * self.pixel_size * n_proj * n_fov * 1e-6
        e_dep = (1 - np.exp(-sample.get_attenuation_coeff(energy) * w)) * energy * ElectronCharge * influx

        return influx, e_dep / (np.pi * (w2 * self.pixel_size * 1e-6) ** 2)
