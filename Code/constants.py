# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 21:20:59 2014

@author: T_ohne_admin
"""

class Constants(object):
    """
    This class is the standard class that is inherited by all subproblems as 
    well as the master problem
    Constants:
        c_w = 4180 J/kgK
        
        dt = 900 s (time step length)
        
        t_amb = 20 °C (storage's surroundings temperature)
        
        k_g --> gas price in €/J
        
        k_el --> electricity price in €/J
        
        r_el --> electricity revenue in €/J
    """
    
    c_w = 4180
    dt = 900
    t_amb = 20
    k_g  = 8.00  / (100 * 3600 * 1000) # €/J
    k_el = 29.21 / (100 * 3600 * 1000) # €/J
    r_el = 5.00  / (100 * 3600 * 1000) # €/J
    max_time_subs = 5
    max_time_master = 10
    
    timesteps = 2 * 24 * 4
    t=0
    
    cg_iterations = 1
    inner_loops = [1,1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    final_max = 5
    
    outer_loops = len(inner_loops)
    outer_max = cg_iterations + len(inner_loops)
    overall_max = cg_iterations + sum(inner_loops)
    subgradient_max = sum(inner_loops)
    
    eps = 0.0
    alpha = 1.0
    
    
    t_bivalent = -2.0
    
    pricing_MIPGap = 0.01
    
    final_MIPGap = 0.01
    
    initial_in_master = 3
    
    pricing_time_limit = 600
    
    plot = False
    
    path = "/home/qwertzuiopu/Data"
    
    random_proposals_in_master = 3
        
class Storage(Constants):
    """
    This class adds the storage's constants to the Constants class
        sto_m: Water mass inside the storage in kg
        
        sto_UA: U*A in W/m2K
        
        T_init: Starting temperature of the storage in °C
        
        T_max: Maximal temperature in the storage in °C
    """
    
    t_init = 0
    t_max  = 0
    sto_m  = 0
    sto_UA = 0
    
class GlobVar():
    
    overall_count = 0
    inner_count = 0
    outer_count = 0
    final_count = 0
    first_iteration = True
    random_proposals_in_master = True