import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp


RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'
import sys
sys.path.append(RFTrackPath)
import RF_Track as RFT


def RF_TO_XSUITE_converter(B0,p0c,beta,m_ion,q0,length):

    

    #p0c=setup.Ions_P*1e6
    context = xo.ContextCpu()
    buf = context.new_buffer()
    #m_ion=part.mass0[0]
    #m_ion=setup.Ions_mass*1e6 
    #q0=setup.Ions_Q 
    length = length #% m, SPS circumference length


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
            mass0=m_ion, q0=q0, p0c=p0c, 
            x=x, px=px, y=y, py=py,
            zeta=zeta, delta=delta)
    
    particles1.at_turn=(S/length)*1e-3
    
    
    return particles1



