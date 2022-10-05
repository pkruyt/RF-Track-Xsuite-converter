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
####################
# Choose a context #
####################

context = xo.ContextCpu()
#context = xo.ContextCpu(omp_num_threads=5)
#context = xo.ContextCupy()
#context = xo.ContextPyopencl('0.0')

buf = context.new_buffer()



n_part = int(1e1)


# Ion properties:
m_u = 931.49410242e6 # eV/c^2 -- atomic mass unit
A = 207.98 # Lead-208
Z = 82  # Number of protons in the ion (Lead)
Ne = 3 # Number of remaining electrons (Lithium-like)
m_e = 0.511e6 # eV/c^2 -- electron mass
m_p = 938.272088e6 # eV/c^2 -- proton mass
c = 299792458.0 # m/s

m_ion = A*m_u + Ne*m_e # eV/c^2

equiv_proton_momentum = 236e9 # eV/c = gamma_p*m_p*v

gamma_p = np.sqrt( 1 + (equiv_proton_momentum/m_p)**2 ) # equvalent gamma for protons in the ring


p0c = equiv_proton_momentum*(Z-Ne) # eV/c
gamma = np.sqrt( 1 + (p0c/m_ion)**2 ) # ion relativistic factor
beta = np.sqrt(1-1/(gamma*gamma)) # ion beta

q0=Z-Ne

#%% 
##################
# RF CAVITY #
##################

fname_sequence = '/home/pkruyt/anaconda3/lib/python3.9/site-packages/xtrack/test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json'

with open(fname_sequence, 'r') as fid:
     input_data = json.load(fid)
     input_line=input_data['line']
     sequence = xt.Line.from_dict(input_line)


sequence = xt.Line.from_dict(input_line)

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

# length=1


# line = xt.Line(
#     elements=[xt.Drift(length=length) #m           
#               ],
#     element_names=['drift_0'])

# tracker = xt.Tracker(_context=context, _buffer=buf, line=line)


# Build a reference particle
particle_sample = xp.Particles(mass0=m_ion, q0=q0, p0c=p0c)

particles0 = xp.generate_matched_gaussian_bunch(
          num_particles=n_part,
          #total_intensity_particles=bunch_intensity,
          nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
          #R_matrix=r_matrix,
          particle_ref=particle_sample,
          tracker=tracker_SPS
          #,steps_r_matrix=steps_r_matrix
          )


#particles0.zeta=0

# zeta_old=[0,0]

# particles0 = xp.Particles(_context=context,
#         mass0=m_ion, q0=q0, p0c=p0c, # 7 TeV
#         x=[1e-3, 0], px=[1e-6, -1e-6], y=[0, 1e-3], py=[2e-6, 0],
#         zeta=zeta_old, delta=[0, 0])

particles_old=particles0.copy()

#particles0.delta=0
print('z0',particles0.zeta)
sequence.particle_ref = particle_sample
twiss = tracker_SPS.twiss(symplectify=False)


num_turns=200
tracker_SPS.track(particles0, num_turns=num_turns, turn_by_turn_monitor=True)
print('z1',particles0.zeta)
zeta0=particles0.zeta


# x=tracker.record_last_track.x
# y=tracker.record_last_track.y
z=tracker_SPS.record_last_track.zeta

#%%

from  Xsuite_to_RF import XSUITE_TO_RF_converter
from RF_to_Xsuite import RF_TO_XSUITE_converter

zeta_init = particles_old.zeta
length = twiss['circumference']
beta=particles_old.beta0[0]

S=num_turns*length

#%%

beam0=XSUITE_TO_RF_converter(particles0,zeta_init,S)
part1=RF_TO_XSUITE_converter(beam0,p0c,beta,m_ion,q0,S)


zeta1=part1.zeta

#%%

#zdelta0=particles0.delta

#zeta_diff=zeta0-zeta1

# zeta_max=max(zeta_diff)
#zeta_rela = (zeta0-zeta1)/zeta0

# beam1=XSUITE_TO_RF_converter(part0,zeta_init,S)
# beam11=beam1.get_phase_space("%x %Px  %y %Py %Z %d %P")