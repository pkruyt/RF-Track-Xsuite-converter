import json
import numpy as np

import time
import xobjects as xo
import xtrack as xt
import xpart as xp


RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'
import sys
sys.path.append(RFTrackPath)
import RF_Track as RFT
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
from scipy import constants 
####################
# Choose a context #
####################

context = xo.ContextCpu()
#context = xo.ContextCpu(omp_num_threads=5)
#context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

buf = context.new_buffer()



import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

## Generate a simple line
line = xt.Line(
    elements=[xt.Drift(length=2.),
              xt.Multipole(knl=[0, 1.], ksl=[0,0]),
              xt.Drift(length=1.),
              xt.Multipole(knl=[0, -1.], ksl=[0,0])],
    element_names=['drift_0', 'quad_0', 'drift_1', 'quad_1'])

## Attach a reference particle to the line (optional)
## (defines the reference mass, charge and energy)
particle_ref = xp.Particles(p0c=6500e9, #eV
                                 q0=1, mass0=xp.PROTON_MASS_EV)

line.particle_ref=particle_ref

## Choose a context
context = xo.ContextCpu()         # For CPU
# context = xo.ContextCupy()      # For CUDA GPUs
# context = xo.ContextPyopencl()  # For OpenCL GPUs

## Transfer lattice on context and compile tracking code
tracker = line.build_tracker(_context=context)

## Build particle object on context
n_part = 200
particles = xp.Particles(p0c=6500e9, #eV
                        q0=1, mass0=xp.PROTON_MASS_EV,
                        x=np.random.uniform(-1e-3, 1e-3, n_part),
                        px=np.random.uniform(-1e-5, 1e-5, n_part),
                        y=np.random.uniform(-2e-3, 2e-3, n_part),
                        py=np.random.uniform(-3e-5, 3e-5, n_part),
                        zeta=np.random.uniform(-1e-2, 1e-2, n_part),
                        delta=np.random.uniform(-1e-4, 1e-4, n_part),
                        _context=context)


particles0=particles.copy()

## Track (saving turn-by-turn data)
n_turns = 100
tracker.track(particles, num_turns=n_turns,
              turn_by_turn_monitor=True)

## Turn-by-turn data is available at:
tracker.record_last_track.x
tracker.record_last_track.px
# etc...

particles1=particles.copy()

#%%

from  Xsuite_to_RF import XSUITE_TO_RF_converter
from RF_to_Xsuite import RF_TO_XSUITE_converter

zeta_init = particles0.zeta
length = line.get_length()
beta=particles0.beta0[0]

S=n_turns*length

#%%

beam0=XSUITE_TO_RF_converter(particles0,zeta_init,S)
part1=RF_TO_XSUITE_converter(beam0,particle_ref,S)


zeta1=part1.zeta



