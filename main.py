# -*- coding: utf-8 -*-

import numpy as np
import os
from simulator import *
from sinogram import *
from instrument import *


sim = Simulator()
sim.read_full_sinogram(os.path.join('test', 'sino_01000.tiff'))