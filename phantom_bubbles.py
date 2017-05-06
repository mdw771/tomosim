from xdesign import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import dxchange
import tomopy
import os

SIZE = 2048

obj_path = 'foam_obj.tiff'
if not os.path.exists(obj_path):
    print('Building...')
    np.random.seed(0) # random seed for repeatability
    p1 = Foam(size_range=[0.05, 0.002])
    d1 = discrete_phantom(p1, SIZE)
    plt.imshow(d1, cmap='viridis')
    dxchange.write_tiff(d1, obj_path, dtype='float32', overwrite=True)
    print('Phantom built.\n')
else:
    print('Reading existing...')
    d1 = dxchange.read_tiff(obj_path)

d1 = d1[np.newaxis, :, :]

theta = tomopy.angles(4096)
print('Radon')
sino = tomopy.project(d1.astype('float32'), theta, center=1024, ncore=10, nchunk=50)
sino = sino / sino.max()
sino = np.exp(-sino)
dxchange.write_tiff(np.squeeze(sino), 'foam_sino')

# p1 = Phantom(shape='circle')
# p1.sprinkle(300, [0.1, 0.03], gap=10, mass_atten=1)
# d1 = discrete_phantom(p1, SIZE)
# plt.imshow(d1, cmap='viridis')
# plt.show()