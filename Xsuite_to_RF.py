import RF_Track as RFT
import sys
import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp


RFTrackPath = '/home/pkruyt/cernbox/rf-track-2.0'
sys.path.append(RFTrackPath)


def XSUITE_TO_RF_converter(particles, zeta_init):
    """The desired variables that are needed for a beam in RF Track are:

        1. X
        2. XP
        3. Y
        4. YP
        5. T
        6. P
        7. M
        8. Q             
        """
    ###########################################################################
    """These parameters are needed to compute to corresponding variables in RF Track"""
    p0c = particles.p0c[0]
    m_ion = particles.mass0
    q0 = particles.q0
    beta=particles.beta0[0]
    n_particles = len(particles.particle_id)

    # length = length #% m, SPS circumference length
    

    p_tot = (particles.delta*p0c+p0c)
    print('p_tot',p_tot)
    Px = particles.px*p0c
    print('Px',Px)
    Py = particles.py*p0c
    print('Py',Py)
    Pz2 = (p_tot)**2-(Px)**2-(Py)**2
    
    Pz = np.sqrt(Pz2)
    print('Pz',Pz)
    
    gamma_particles = np.sqrt(1 + (Pz/m_ion)**2)  # ion relativistic factor
    # gamma_particles=np.sqrt( 1 + (p_tot/m_ion)**2 ) # ion relativistic factor
    beta_particles = np.sqrt(1-1/(gamma_particles*gamma_particles))  # ion beta
        
    #accumulated_length = (particles.at_turn)*length
    #accumulated_length = num_turns*length
    accumulated_length = particles.s
    print('accumulated_length',accumulated_length)
    ###########################################################################
    """Direct calculation of the corresponding variables in RF Track:"""

    # X
    X = particles.x * 1e3 #mm
    # XP
    ratio_x = Px/Pz
    #print('ratio_x',ratio_x)
    angle_x = np.arctan(ratio_x)*1e3
    #print('angle_x',angle_x)
    # Y
    Y = particles.y * 1e3 #mm
    # YP
    ratio_y = Py/Pz
    angle_y = np.arctan(ratio_y)*1e3
    # T
    c=299792458.0
    t_tot = (accumulated_length-zeta_init)/(beta_particles) #arrival time in m/c
    
    print('beta_particles',beta_particles)
    
    t = (t_tot)*1e3 #mm/c
    print('t',t)
    # P
    P = p_tot*1e-6 #Mev/c
    # M
    mass = m_ion*np.ones(n_particles)*1e-6 #Mev/c^2
    # Q
    q = q0*np.ones(n_particles) #e
    ###########################################################################
    """Combine into one array and build beam in RF Track"""
    
    t_ref=accumulated_length*1e3/beta
    t_tot = (particles.zeta)/(beta-beta_particles)/c #arrival time in m/c
    arr_ref = np.column_stack(((0, 0, 0, 0, t_ref, p0c*1e-6, mass, q)))
    arr = np.column_stack(((X, angle_x, Y, angle_y, t, P, mass, q)))
    
    arr=np.vstack([arr_ref,arr])
    beam = RFT.Bunch6d(arr)
    
    #I = beam.get_info()
    #I.S = 10



    return beam
