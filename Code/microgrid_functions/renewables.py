# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 10:47:42 2014

@author: T_ohne_admin
"""

import math
import numpy as np

# Compute the sum of available PV and wind energy

def compute_renewables(wind_speed, 		# wind_speed --> Array of the current wind speed forecast in Watt
		       solar_irradiation,	# solar_irradiation --> Array of the current solar_irradiation forecast in Watt
		       hp_number,
		       chp_number):	

    wind_cut_in = 2.0			#2		wind_cut_in  --> Cut in speed in m/s
    wind_cut_out = 25.0			#25		wind_cut_out --> Cut out speed in m/s
    wind_nominal = 11.0			#17		wind_nominal --> Nominal wind speed in m/s
    solar_area = 20.0			#145		solar_area   --> Area of installed PV modules in m2 / House
    solar_efficiency = 0.22		#0.22		solar_efficiency --> Efficiency of the PV modules
    diameter_rotor = 12.0
    A_wind = 0.25 * math.pi * diameter_rotor**2
    number_wind_turbines = 5.0
    eta_wind = 0.42
    
    # PV production
    solar = (hp_number + chp_number) * solar_area * solar_efficiency * solar_irradiation
        
    # Compute wind energy production
    wind = np.zeros_like(wind_speed)
    
    index_nominal = (wind_speed >= wind_nominal) * (wind_speed < wind_cut_out)
    wind[index_nominal] = 0.5 * 1 * wind_nominal**3 * A_wind * eta_wind * number_wind_turbines
    
    index_partial_load = (wind_speed >= wind_cut_in) * (wind_speed < wind_nominal)
    wind[index_partial_load] = 0.5 * 1 * wind_speed[index_partial_load]**3 * A_wind * eta_wind * number_wind_turbines
    
    return solar+wind