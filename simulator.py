# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import dxchange
import os
from instrument import *
from sinogram import *
from util import *


class Simulator(object):

    def __init__(self):

        self.sinos_local = []
        self.sinos_tomosaic = []
        self.raw_sino = None
        self.full_recon_local = None
        self.stitched_sino_tomosaic = None

    def read_raw_sinogram(self, fname, type='tiff', center=None, **kwargs):

        if type == 'hdf5':
            slice = kwargs['slice']
            raw_sino = np.squeeze(dxchange.read_aps_32id(fname, sino=(slice, slice+1)))
        else:
            raw_sino = dxchange.read_tiff(fname)
        self.raw_sino = Sinogram(raw_sino, 'raw', coords=center, center=center)

    def raw_sino_add_noise(self, fraction_mean=0.01):

        self.raw_sino.add_poisson_noise(fraction_mean=fraction_mean)

    def load_instrument(self, instrument):

        assert isinstance(instrument, Instrument)
        self.inst = instrument

    def sample_full_sinogram_localtomo(self):

        for center_coords in self.inst.center_positions:

            print('Sampling sinogram for center ({:d}, {:d}).'.format(center_coords[0], center_coords[1]))

            y0, x0 = center_coords
            sino = np.zeros(self.raw_sino.shape)
            w = sino.shape[1]
            nang = sino.shape[0]

            # compute trajectory of center of FOV in sinogram space
            a = w - y0 - x0 - 1
            b = np.pi / nang
            c = x0
            ylist = np.arange(nang, dtype='int')
            xlist = np.round(a * np.sin(b * ylist) + c)

            mask = np.zeros(sino.shape, dtype='bool')
            dx2 = int(self.inst.fov / 2)
            for (y, x) in np.dstack([ylist, xlist])[0].astype('int'):
                endl = int(x - dx2) if x - dx2 >= 0 else 0
                endr = int(endl + self.inst.fov) if (endl + self.inst.fov <= w) else w
                mask[int(y), endl:endr] = True

            sino[mask] = self.raw_sino.sinogram[mask]
            local_sino = Sinogram(sino, 'local', coords=(y0, x0))
            self.sinos_local.append(local_sino)

    def recon_all_local(self, save_path=None):

        for sino in self.sinos_local:
            print('Reconstructing local tomograph at ({:d}, {:d}).'.format(sino.coords[0], sino.coords[1]))
            sino.reconstruct(add_mask=True, fov=self.inst.fov)
            if save_path is not None:
                dxchange.write_tiff(sino.recon * sino.recon_mask, os.path.join(save_path, 'recon_loc_{:d}_{:d}'.
                                                                               format(sino.coords[0], sino.coords[1])))

    def stitch_all_recons_local(self):

        self.full_recon_local = np.zeros(self.raw_sino.shape)
        for sino in self.sinos_local:
            self.full_recon_local[sino.recon_mask] = sino.recon[sino.recon_mask]
        return self.full_recon_local

    def sample_full_sinogram_tomosaic(self):

        for center_pos in self.inst.stage_positions:

            sino = np.zeros(self.raw_sino.shape)
            w = sino.shape[1]
            dx2 = int(self.inst.fov / 2)

            endl = center_pos - dx2 if center_pos - dx2 >= 0 else 0
            endr = endl + self.inst.fov if (endl + self.inst.fov <= w) else w

            partial_sino = self.raw_sino.sinogram[:, endl:endr]
            partial_sino = Sinogram(partial_sino, 'tomosaic', coords=center_pos, center=self.raw_sino.center)
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

    def recon_full_tomosaic(self):

        fov = self.stitched_sino_tomosaic.sinogram.shape[1]
        print('Reconstructing full tomosaic sinogram.')
        self.stitched_sino_tomosaic.reconstruct(add_mask=True, fov=fov)
        self.full_recon_tomosaic = self.stitched_sino_tomosaic.recon
        return self.full_recon_tomosaic

