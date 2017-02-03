import numpy as np
import tomopy


class Sinogram(object):

    def __init__(self, type, sinogram, coords):

        assert type in ('local', 'tomosaic')

        self.type = type
        self.sinogram = sinogram
        self.coords = coords
        self.recon = None
        self.mask = None

    def reconstruct(self, fov):

        nang = self.sinogram.shape[0]
        theta = tomopy.angles(nang)
        if type == 'local':
            center = self.coords[1]
        elif type == 'tomosaic':
            center = self.coords
        else:
            center = None
        rec = tomopy.recon(self.sinogram[:, np.newaxis, :], theta, center=center)
        rec = np.squeeze(rec)

        self.recon = rec
        radius = fov / 2.
        y, x = np.ogrid[0.5:0.5+rec.shape[0], 0.5:0.5+rec.shape[1]]
        self.mask = (y - self.coords[0]) ** 2 + (x - self.coords[1]) ** 2 < radius ** 2

