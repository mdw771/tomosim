import numpy as np
import dxchange
from instrument import *
from sinogram import *


class Simulator(object):

    def __init__(self):

        self.sinos_local = []
        self.sinos_tomosaic = []

    def read_full_sinogram(self, fname, type='tiff', **kwargs):

        if type == 'tiff':
            self.full_sino = dxchange.read_tiff(fname)
        elif type == 'hdf5':
            slice = kwargs['slice']
            self.full_sino = np.squeeze(dxchange.read_aps_32id(fname, sino=(slice, slice+1)))

    def load_instrument(self, instrument):

        assert isinstance(instrument, Instrument)
        self.inst = instrument

    def sample_full_sinogram_localtomo(self):

        for center_coords in self.inst.center_positions:

            y0, x0 = center_coords
            sino = np.zeros(self.full_sino.shape)
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
            for (y, x) in np.dstack([ylist, xlist])[0]:
                endl = x - dx2 if (x - dx2 >= 0) else 0
                endr = endl + self.inst.fov if (endl + self.inst.fov <= w) else w
                mask[y, endl:endr] = True

            sino[mask] = self.full_sino[mask]
            local_sino = Sinogram('local', sino, (y0, x0))
            self.sinos_local.append(local_sino)

    def sample_full_sinogram_tomosaic(self):

        for center_pos in self.inst.stage_positions:

            sino = np.zeros(self.full_sino.shape)
            w = sino.shape[1]
            dx2 = int(self.inst.fov / 2)

            endl = center_pos - dx2 if (center_pos - dx2 >= 0) else 0
            endr = endl + self.inst.fov if (endl + self.inst.fov <= w) else w
            sino[:, endl:endr] = self.full_sino[:, endl:endr]

            partial_sino = Sinogram('tomosaic', sino, center_pos)
            self.sinos_tomosaic.append(partial_sino)

    def recon_all_local(self):

        for sino in self.sinos_local:
            sino.reconstruct()

    def stitch_all_local(self):

        pass