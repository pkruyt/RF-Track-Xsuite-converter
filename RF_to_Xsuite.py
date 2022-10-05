import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp


RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'
import sys
sys.path.append(RFTrackPath)
import RF_Track as RFT


def RF_TO_XSUITE_converter(B0,p0c,beta,m_ion,q0,S):
    """The desired variables that are needed for a beam in Xsuite are:
        
        1. x
        2. px
        3. y
        4. py
        5. zeta
        6. delta         
        """
    ########################################################################### 
    """These parameters are needed to compute to corresponding variables in RF Track"""
    

    #p0c=setup.Ions_P*1e6
    context = xo.ContextCpu()
    buf = context.new_buffer()
        
    beam=B0.get_phase_space("%x %Px  %y %Py %t %P")
    
    ###########################################################################
    #x
    x = beam[:,0]*1e-3
    #px
    Px = beam[:,1]
    px = Px/p0c*1e6
    #y
    y = beam[:,2]*1e-3
    #py
    Py = beam[:,3]
    py = Py/p0c*1e6
    #z
    t = beam[:,4]
    accumulated_length = [S]*len(x) 
    zeta=accumulated_length-(beta*t*1e-3) # according to definition from https://github.com/xsuite/xsuite/issues/8
    #delta        
    P = beam[:,5]*1e6
    delta = (P-p0c)/p0c
    ###########################################################################
    """Build into one beam in Xsuite"""

    particles = xp.Particles(_context=context,
            mass0=m_ion, q0=q0, p0c=p0c, 
            x=x, px=px, y=y, py=py,
            zeta=zeta, delta=delta)
    
    #particles1.at_turn=(S/length)*1e-3
    
    return particles



