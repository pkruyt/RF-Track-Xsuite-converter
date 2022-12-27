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



drift=xt.Drift(length=1)

line=xt.Line()

line.append_element(drift, 'drift')




tracker=xt.Tracker(_context=context, _buffer=buf, line=line,reset_s_at_end_turn=False)

tracker.track(particles,num_turns=1)


#tracker = line.build_tracker(_context=context, )

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


#beta=particles.beta0[0]


#%%

def test_inplace():

    particles = xp.Particles(p0c=1e15, #eV
                            q0=1, mass0=xp.PROTON_MASS_EV,
                            x=[-1e-3],
                            px=[-1e-5],
                            y=[2e-3],
                            py=[-3e-5],
                            zeta=[-1e-2],
                            delta=[1],
                            _context=context)
    
    
    beam_RF=XSUITE_TO_RF_converter(particles)
    particles_XS=RF_TO_XSUITE_converter(beam_RF)
    particles_XS=particles_XS.filter(particles_XS.x!=0)
      
    
    
    phase0=Xsuite_get_phase_space(particles)
       
    phase1=Xsuite_get_phase_space(particles_XS)
    
    assert np.allclose(phase1,phase0,rtol=1e-9, atol=1e-9)
    
    
    


test_inplace()

