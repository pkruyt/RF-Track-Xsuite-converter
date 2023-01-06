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



drift=xt.Drift(length=1000)

line=xt.Line()

line.append_element(drift, 'drift')




tracker=xt.Tracker(_context=context, _buffer=buf, line=line,reset_s_at_end_turn=False)


D=RFT.Drift(length_=1000)
L=RFT.Lattice()

L.append(D)

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



#%%

def test_inplace_XS():

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
    
    
    
def test_inplace_RF():
    init = [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.00000000e+09, 9.38272088e+02,  1.00000000e+00,  1.00000000e+00],
            [-1.00000000e+00, -5.00000000e-03,  2.00000000e+00,-1.50000000e-02,  1.00000000e+01,  2.00000000e+09,9.38272088e+02,  1.00000000e+00,  1.00000000e+00]]


    init = [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,  0.00000000e+00,  1.00000000e+09, 9.38272088e+02,  1.00000000e+00],
            [-1.00000000e+00, -5.00000000e-03,  2.00000000e+00,-1.50000000e-02,  1.00000000e+01,  2.00000000e+09,9.38272088e+02,  1.00000000e+00]]

    init_array = np.array(init)
    
    
    beam_RF=RFT.Bunch6d(init_array)



     
    #convert back to Xsuite so they can be compared
    particles_XS=RF_TO_XSUITE_converter(beam_RF)
    particles_XS=particles_XS.filter(particles_XS.x!=0)
    
    beam_RF2=XSUITE_TO_RF_converter(particles_XS)
      

    #read out both phase spaces
    phase0=beam_RF.get_phase_space("%x %XP  %y %YP %t %P %m %Q")
    phase1=beam_RF2.get_phase_space("%x %XP  %y %YP %t %P %m %Q")
    
    assert np.allclose(phase1,phase0,rtol=1e-9, atol=1e-9)
    return phase0,phase1
    
#%%

def test_drift():

    particles = xp.Particles(p0c=1e15, #eV
                            q0=1, mass0=xp.PROTON_MASS_EV,
                            x=[-1e-3,1e-3],
                            px=[-1e-5,1e-5],
                            y=[2e-3,-2e-3],
                            py=[-3e-5,3e-5],
                            zeta=[-1e-2,1e-2],
                            delta=[1,-0.1],
                            _context=context)
     
    particles_init = particles.copy() # save for later use


    #tracking in Xsuite
    tracker.track(particles,num_turns=1)


    #Tracking in RF Track
    beam_RF=XSUITE_TO_RF_converter(particles_init)
    beam_RF=L.track(beam_RF)

  
    #convert back to Xsuite so they can be compared
    particles_XS=RF_TO_XSUITE_converter(beam_RF)
    particles_XS=particles_XS.filter(particles_XS.x!=0)
      

    #read out both phase spaces
    phase0=Xsuite_get_phase_space(particles)
    phase1=Xsuite_get_phase_space(particles_XS)
    
    assert np.allclose(phase1,phase0,rtol=1e-9, atol=1e-9)




    
def test_double_drift():

    particles = xp.Particles(p0c=1e15, #eV
                             q0=1, mass0=xp.PROTON_MASS_EV,
                             x=[-1e-3],
                             px=[-1e-5],
                             y=[2e-3],
                             py=[-3e-5],
                             zeta=[-1e-2],
                             delta=[1],
                             _context=context)
     
    particles_init = particles.copy() # save for later use


    #tracking in Xsuite
    tracker.track(particles,num_turns=2)


    #Tracking in RF Track
    beam_RF=XSUITE_TO_RF_converter(particles_init)
    beam_RF=L.track(beam_RF)
    beam_RF=L.track(beam_RF)

  
    #convert back to Xsuite so they can be compared
    particles_XS=RF_TO_XSUITE_converter(beam_RF)
    particles_XS=particles_XS.filter(particles_XS.x!=0)
      

    #read out both phase spaces
    phase0=Xsuite_get_phase_space(particles)
    phase1=Xsuite_get_phase_space(particles_XS)
    
    assert np.allclose(phase1,phase0,rtol=1e-9, atol=1e-9)    




def test_beam_with_two_particles():
    particles = xp.Particles(p0c=1e15, #eV
                            q0=1, mass0=xp.PROTON_MASS_EV,
                            x=[-1e-3,1e-3],
                            px=[-1e-5,1e-5],
                            y=[2e-3,-2e-3],
                            py=[-3e-5,3e-5],
                            zeta=[-1e-2,1e-2],
                            delta=[1,-0.1],
                            _context=context)
     
    particles_init = particles.copy() # save for later use


    #tracking in Xsuite
    tracker.track(particles,num_turns=1)


    #Tracking in RF Track
    beam_RF=XSUITE_TO_RF_converter(particles_init)
    beam_RF=L.track(beam_RF)

  
    #convert back to Xsuite so they can be compared
    particles_XS=RF_TO_XSUITE_converter(beam_RF)
    particles_XS=particles_XS.filter(particles_XS.x!=0)
      

    #read out both phase spaces
    phase0=Xsuite_get_phase_space(particles)
    phase1=Xsuite_get_phase_space(particles_XS)
    
    assert np.allclose(phase1,phase0,rtol=1e-9, atol=1e-9)



def test_beam_with_list_of_random_particles():
   num_part=4000

   np.random.seed(123)

   particles = xp.Particles(p0c=1e15, #eV
                           q0=1, mass0=xp.PROTON_MASS_EV,
                           x=np.random.normal(-1e-3, 1e-3, num_part),
                           px=np.random.normal(-1e-3, 1e-3, num_part),
                           y=np.random.normal(-1e-3, 1e-3, num_part),
                           py=np.random.normal(-1e-3, 1e-3, num_part),
                           zeta=np.random.normal(-1e-3, 1e-3, num_part),
                           delta=np.random.normal(-1e-3, 1e-3, num_part),
                           _context=context)
    
   particles_init = particles.copy() # save for later use


   #tracking in Xsuite
   #tracker.track(particles,num_turns=1)


   #Tracking in RF Track
   beam_RF=XSUITE_TO_RF_converter(particles_init)
   #beam_RF=L.track(beam_RF)


   #convert back to Xsuite so they can be compared
   particles_XS=RF_TO_XSUITE_converter(beam_RF)
   particles_XS=particles_XS.filter(particles_XS.x!=0)
     

   #read out both phase spaces
   phase0=Xsuite_get_phase_space(particles)
   phase1=Xsuite_get_phase_space(particles_XS)

   assert np.allclose(phase1,phase0,rtol=1e-7, atol=1e-7)


num_part=4000

np.random.seed(123)

particles = xp.Particles(p0c=1e15, #eV
                        q0=1, mass0=xp.PROTON_MASS_EV,
                        x=np.random.normal(-1e-3, 1e-3, num_part),
                        px=np.random.normal(-1e-3, 1e-3, num_part),
                        y=np.random.normal(-1e-3, 1e-3, num_part),
                        py=np.random.normal(-1e-3, 1e-3, num_part),
                        zeta=np.random.normal(-1e-3, 1e-3, num_part),
                        delta=np.random.normal(-1e-3, 1e-3, num_part),
                        _context=context)
 
particles_init = particles.copy() # save for later use


#tracking in Xsuite
tracker.track(particles,num_turns=1)


#Tracking in RF Track
beam_RF=XSUITE_TO_RF_converter(particles_init)
beam_RF=L.track(beam_RF)


#convert back to Xsuite so they can be compared
particles_XS=RF_TO_XSUITE_converter(beam_RF)
particles_XS=particles_XS.filter(particles_XS.x!=0)
  

#read out both phase spaces
phase0=Xsuite_get_phase_space(particles)
phase1=Xsuite_get_phase_space(particles_XS)

# p0=phase0[3]
# p1=phase1[3]

assert np.allclose(phase1,phase0,rtol=1e-7, atol=1e-7)

def test_beam_with_list_of_random_particles_drift():

    num_part=4000
    
    np.random.seed(123)
    
    particles = xp.Particles(p0c=1e15, #eV
                            q0=1, mass0=xp.PROTON_MASS_EV,
                            x=np.random.normal(-1e-3, 1e-3, num_part),
                            px=np.random.normal(-1e-3, 1e-3, num_part),
                            y=np.random.normal(-1e-3, 1e-3, num_part),
                            py=np.random.normal(-1e-3, 1e-3, num_part),
                            zeta=np.random.normal(-1e-3, 1e-3, num_part),
                            delta=np.random.normal(-1e-3, 1e-3, num_part),
                            _context=context)
     
    particles_init = particles.copy() # save for later use
    
    
    #tracking in Xsuite
    tracker.track(particles,num_turns=1)
    
    
    #Tracking in RF Track
    beam_RF=XSUITE_TO_RF_converter(particles_init)
    beam_RF=L.track(beam_RF)
    
    
    #convert back to Xsuite so they can be compared
    particles_XS=RF_TO_XSUITE_converter(beam_RF)
    particles_XS=particles_XS.filter(particles_XS.x!=0)
      
    
    #read out both phase spaces
    phase0=Xsuite_get_phase_space(particles)
    phase1=Xsuite_get_phase_space(particles_XS)
    
    # p0=phase0[3]
    # p1=phase1[3]
    
    assert np.allclose(phase1,phase0,rtol=1e-7, atol=1e-7)

test_inplace_XS()
test_inplace_RF()
test_drift()
test_double_drift()
test_beam_with_two_particles()
test_beam_with_list_of_random_particles()
# test_beam_with_list_of_random_particles_drift()
print('\nno errors: \nall tests have passed')







    



