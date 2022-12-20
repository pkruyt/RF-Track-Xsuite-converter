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



n_part = int(1e5)


# Ion properties:
A = 207.98 # Lead-208
Z = 82  # Number of protons in the ion (Lead)
Ne = 3 # Number of remaining electrons (Lithium-like)

c = constants.c # m/s
m_u=constants.physical_constants['atomic mass unit-electron volt relationship'][0] # eV/c^2 -- atomic mass unit
m_e=constants.m_e/constants.e*c*c # eV/c^2 -- electron mass
m_p=constants.m_p/constants.e*c*c # eV/c^2 -- proton mass

m_ion = A*m_u + Ne*m_e # eV/c^2

equiv_proton_momentum = 236e9 # eV/c = gamma_p*m_p*v

gamma_p = np.sqrt( 1 + (equiv_proton_momentum/m_p)**2 ) # equvalent gamma for protons in the ring


p0c = equiv_proton_momentum*(Z-Ne) # eV/c
gamma = np.sqrt( 1 + (p0c/m_ion)**2 ) # ion relativistic factor
beta = np.sqrt(1-1/(gamma*gamma)) # ion beta

#%% 
##################
# RF CAVITY #
##################

fname_sequence = fname_sequence ='/home/pkruyt/cernbox/xsuite/xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json'

with open(fname_sequence, 'r') as fid:
     input_data = json.load(fid)
     input_line=input_data['line']
     sequence = xt.Line.from_dict(input_line)





sequence = xt.Line.from_dict(input_line)



sps_voltage = 6*(1e5)
sps_freq=200*(1e6)


rf = xt.Cavity(voltage=sps_voltage,
               frequency=sps_freq
               )


# sequence.append_element(rf,'cavity')

#%% 
##################
# Laser Cooler #
##################

#sigma_dp = 2e-4 # relative ion momentum spread

#bunch_intensity = 1e11
sigma_z = 22.5e-2
nemitt_x = 2e-6
nemitt_y = 2.5e-6

sigma_dp = sigma_z / beta
sigma_dp = 2e-4 # relative ion momentum spread




#%%
##################
# Build TrackJob #
##################



tracker_SPS = xt.Tracker(_context=context, _buffer=buf, line=sequence)


# Build a reference particle
particle_sample = xp.Particles(mass0=m_ion, q0=Z-Ne, p0c=p0c)

particles0 = xp.generate_matched_gaussian_bunch(
         num_particles=n_part,
         #total_intensity_particles=bunch_intensity,
         nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
         #R_matrix=r_matrix,
         particle_ref=particle_sample,
         tracker=tracker_SPS
         #,steps_r_matrix=steps_r_matrix
         )


#%%
input_line2=input_line.copy()

input_element_names = input_line2['element_names']
del input_element_names[5405]


input_elements = input_line2['elements']
del input_elements[5405]

sequence_no_rf=xt.Line.from_dict(input_line2)

tracker_SPS_no_rf = xt.Tracker(_context=context, _buffer=buf, line=sequence_no_rf)

#%%


sequence.particle_ref = particle_sample
twiss = tracker_SPS.twiss(symplectify=False)

del twiss['particle_on_co']

import pickle

with open('twiss.pkl', 'wb') as f:
    pickle.dump(twiss, f)

#print(twiss)


particles_old=particles0.copy()

zeta_init = particles_old.zeta

 
#%%

#################converter

import RF_Track as RFT

def XSUITE_TO_RF_converter(part):

   p0c=part.p0c[0]
   
   m_ion=part.mass0
   q0=part.q0
   n_part=len(part.particle_id)
   
   
   zeta_old = zeta_init #m
   
   #zeta_old=np.zeros(10000,)
   length = 6911.5038 #% m, SPS circumference length
   Nions = n_part
   beta=part.beta0[0]
   
    
   if len(zeta_old)==len(part.x):
         zeta_old2 = zeta_old
   if len(zeta_old)<len(part.x):
         zeta_old2 = np.insert(zeta_old, 0, 0, axis=0)
   
   #zeta_old2 = zeta_old-zeta_max
   zeta_old2 = zeta_old
   
   #t_first_particle = 
    
   x=part.x *1e3
   y=part.y *1e3
   
   
   p_tot=(part.delta*p0c+p0c)
   
   Px=part.px*p0c
   Py=part.py*p0c
   Pz2=(p_tot)**2-(Px)**2-(Py)**2
   Pz=np.sqrt(Pz2)
   
   
   gamma_part=np.sqrt( 1 + (Pz/m_ion)**2 ) # ion relativistic factor
   #gamma_part=np.sqrt( 1 + (p_tot/m_ion)**2 ) # ion relativistic factor
   
   
   beta_part = np.sqrt(1-1/(gamma_part*gamma_part)) # ion beta
   
   
   ratio_x=Px/Pz
   ratio_y=Py/Pz
   
   angle_x=np.arctan(ratio_x)*1e3
   angle_y=np.arctan(ratio_y)*1e3
   
   #accumulated_length = (part.at_turn)*length
   accumulated_length = (part.at_turn)*length
   print('turn',part.at_turn)
   # print("ACCUMULATED LENGTH",accumulated_length)
   # print("BETA_PART",beta_part)
   # print("BETA_REF",beta)
   # print("DISTANCE",(accumulated_length-zeta_old2))
   
   t_tot=(accumulated_length-zeta_old2)/(beta_part)
   #t_first=max(accumulated_length-zeta_old2)/(beta_part)
   #print('T_FIRST',t_first)
   t=(t_tot)*1e3
   
   
   
   
   #t=t+t_init
   
   
   mass=m_ion*np.ones(n_part)*1e-6
   q=q0*np.ones(n_part)
   
      
   arr=np.column_stack(((  x, angle_x, y ,angle_y , t ,p_tot*1e-6  , mass, q )))
   
   if len(part.x)==Nions:
   
         newrow = [0, 0, 0 ,0 , accumulated_length[0]*1e3/beta ,p0c*1e-6 , mass[0], q[0]]
         arr2 = np.vstack([newrow,arr])

         beam1 = RFT.Bunch6d(arr2)
   else:
        beam1 = RFT.Bunch6d(arr)
   beam1 = RFT.Bunch6d(arr)
       
   
   return beam1


def RF_TO_XSUITE_converter(B0):

    

    #p0c=setup.Ions_P*1e6
    context = xo.ContextCpu()
    buf = context.new_buffer()
    #m_ion=part.mass0[0]
    #m_ion=setup.Ions_mass*1e6 
    #q0=setup.Ions_Q 
    length = 6911.5038 #% m, SPS circumference length


    beam22=B0.get_phase_space("%x %Px  %y %Py %Z %d %P")
    beam222=B0.get_phase_space("%S %t %Vz")
       
    x = beam22[:,0]*1e-3
    Px = beam22[:,1]
    px = Px/p0c*1e6
    
    y = beam22[:,2]*1e-3
    Py = beam22[:,3]
    py = Py/p0c*1e6
        
    Pz = beam222[:,2]
    P = beam22[:,6]*1e6
    #delta = beam22[:,5]*1e-3
    delta = (P-p0c)/p0c
    
    
    
    zeta = beam22[:,4]*1e-3
    S = beam222[:,0]
    t = beam222[:,1]
    Vz = beam222[:,2]
    t_ref = S/beta        
    
    t_diff=t_ref-t
    zeta=t_diff/beta*1e-3
    print('VZ:',Vz[0])
    print('beta',beta)
    
    particles1 = xp.Particles(_context=context,
            mass0=m_ion, q0=Z-Ne, p0c=p0c, 
            x=x, px=px, y=y, py=py,
            zeta=zeta, delta=delta)
    
    particles1.at_turn=(S/length)*1e-3
    
    
    
    
    return particles1


#%%

# beam0_old=XSUITE_TO_RF_converter(particles_old)
# beam0_old=beam0_old.get_phase_space("%x %px %y %py %t %P %m %Q")

#%%




#tracker_SPS_no_rf.track(particles0, num_turns=1, turn_by_turn_monitor=True)


x0=particles0.x
px0=particles0.px
y0=particles0.y
py0=particles0.py
zeta0=particles0.zeta
delta0=particles0.delta


    




#%%

from SPS_lattice import SPS_Lattice

SPS=SPS_Lattice()

beam0_RF=XSUITE_TO_RF_converter(particles_old)

#beam1=SPS.track(beam0_RF)

beam1 = beam0_RF

particles1=RF_TO_XSUITE_converter(beam1)

x1=particles1.x
px1=particles1.px
y1=particles1.y
py1=particles1.py
zeta1=particles1.zeta
delta1=particles1.delta

#%%


zeta_ratio=zeta0[0]/zeta1[0]
