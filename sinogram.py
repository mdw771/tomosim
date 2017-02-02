import numpy as np
import tomopy


class Sinogram(object):

    def __init__(self, type, sinogram, coords):

        assert type in ('local', 'tomosaic')

        self.type = type
        self.sinogram = sinogram
        self.coords = coords

    def reconstruct(self):

        nang = self.sinogram.shape[0]
        theta = tomopy.angles(nang)
        if type == 'local':
            center = self.coords[1]
        elif type == 'tomosaic':
            center = self.coords
        else:
            center = None
        rec = tomopy.recon(self.sinogram[:, np.newaxis, :], theta, center=center)
        self.recon = np.squeeze(rec)