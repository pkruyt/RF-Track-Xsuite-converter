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



## Attach a reference particle to the line (optional)
## (defines the reference mass, charge and energy)
particle_ref = xp.Particles(p0c=6500e9, #eV
                                 q0=1, mass0=xp.PROTON_MASS_EV)



## Choose a context
context = xo.ContextCpu()         # For CPU
# context = xo.ContextCupy()      # For CUDA GPUs
# context = xo.ContextPyopencl()  # For OpenCL GPUs

## Transfer lattice on context and compile tracking code

## Build particle object on context


particles = xp.Particles(p0c=1e15, #eV
                        q0=1, mass0=xp.PROTON_MASS_EV,
                        x=[-1e-3],
                        px=[-1e-5],
                        y=[2e-3],
                        py=[-3e-5],
                        zeta=[-1e-2],
                        delta=[1],
                        _context=context)



#%%
def Xsuite_get_phase_space(particles):

    x=particles.x
    px=particles.px
    y=particles.y
    py=particles.py
    zeta=particles.zeta
    delta=particles.delta
    
    arrays=[x,px,y,py,zeta,delta]
    
    phase_space0_XS=np.vstack(arrays)
    phase_space0_XS=phase_space0_XS.swapaxes(0, 1)
    
    return phase_space0_XS


from  Xsuite_to_RF import XSUITE_TO_RF_converter
from RF_to_Xsuite import RF_TO_XSUITE_converter

zeta_init = particles.zeta
length = 0
beta=particles.beta0[0]

S=0

#%%

beam0=XSUITE_TO_RF_converter(particles,zeta_init)
part1=RF_TO_XSUITE_converter(beam0)


zeta1=part1.zeta

phase_space0_RF=beam0.get_phase_space("%x %Px  %y %Py %Z %d")



phase0=Xsuite_get_phase_space(particles)
phase1=Xsuite_get_phase_space(part1)

part1.p0c


