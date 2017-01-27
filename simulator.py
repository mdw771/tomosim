import numpy as np
import dxchange


class Simulator(object):

    def __init__(self):
        pass

    def read_sinogram(self, fname, type='tiff', **kwargs):

        if type == 'tiff':
            self.sino = dxchange.read_tiff(fname)
        elif type == 'hdf5':
            slice = kwargs['slice']
            self.sino = np.squeeze(dxchange.read_aps_32id(fname, sino=(slice, slice+1)))

