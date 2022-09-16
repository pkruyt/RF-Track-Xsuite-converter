import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp


RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'
import sys
sys.path.append(RFTrackPath)
import RF_Track as RFT


def XSUITE_TO_RF_converter(part,zeta_init,length):

   p0c=part.p0c[0]
   
   m_ion=part.mass0
   q0=part.q0
   n_part=len(part.particle_id)
   
   length = length #% m, SPS circumference length
   beta=part.beta0[0]
   
    
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
   
   
   accumulated_length = (part.at_turn)*length
   
   t_tot=(accumulated_length-zeta_init)/(beta_part)
   t=(t_tot)*1e3
   
   
   mass=m_ion*np.ones(n_part)*1e-6
   q=q0*np.ones(n_part)
   
      
   arr=np.column_stack(((  x, angle_x, y ,angle_y , t ,p_tot*1e-6  , mass, q )))
   
   beam1 = RFT.Bunch6d(arr)
      
   
   return beam1



