# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 15:03:32 2014

@author: tsz

This module contains a few handy functions:
    - Computation of the flow temperature according to a given heating curve
    - Interpolation in a given device characteristic curve
"""

import numpy as np
import mathfunctions as mf

def heatingCurve(ambient_temp, m=0.33, set_room=293.15, set_ambient=263.15, set_flow=273.15+55, set_return=273.15+45):
    """ This function is a straight-forward implementation of the heatingCurve 
        algorithm of our Modelica library (Cities.Supply.BaseClasses.heatingCurve)
    The parameters are:
        ambient_temperature: Temperature time series in K 
            (Note: ambient_temperature should be a 1-dimensional numpy-array)
            
        m:                   Heater characteristic. Normally: 0.25 <= m <= 0.40 

        set_room:            Room's set temperature in K

        set_ambient:         Nominal ambient temperature in K

        set_flow:            Nominal heater flow temperature in K

        set_return:          Nominal heater return temperature in K
    """

    # Determine design room excess temperature
    dTmN = (set_flow + set_return)/2 - set_room
        
    # Calculate design temperature spread of heating system
    dTN  = set_flow - set_return   

    # Compute load situation of heating system (parameter phi)
    phi = np.zeros_like(ambient_temp)
    phi[ambient_temp <= set_room] = (set_room - ambient_temp[ambient_temp <= set_room]) / (set_room - set_ambient)

    # Compute flow temperature according to heating curve    
    flow_temp = np.power(phi, 1/(1 + m))*dTmN + 0.5*phi*dTN + set_room
    
    return flow_temp
    
# How to use this function
#tamb = 273.15 + 30 * (np.random.rand((100))-0.5)
#tflow = heatingCurve(tamb)

def interpolateQ(t_amb, t_amb_set, q_set):
    """ Compute nominal heat outputs for ambient temperature levels that are 
        not equal to the ones defined for the device's characteristics.
        Example: Heat pump manufacturers publish the heat output at 2 °C and 
                 7 °C ambient temperature, but the current outside temperature
                 is 4.5 °C.
    Parameters:
        t_amb:      List/Array holding the outside temperatures in °C

        t_amb_set:  List/Array of temperature levels defined in the 
                        characteristic curves in °C

        q_set:      List/Array with heat output defined in the characteristic
                        curves in W
    """

    # Initialize result array
    q_result = np.zeros_like(t_amb)
    
    for i in range(np.size(t_amb)):
        # Determine the right interval for the interpolation
        j = 1
        while (j < np.size(t_amb_set)-1) and (t_amb[i] > t_amb_set[j]):
            j = j + 1
            
        # Interpolate the heat output at the given ambient temperature
        q_result[i] = mf.interpolation(t_amb[i], [t_amb_set[j-1], t_amb_set[j]], [q_set[j-1], q_set[j]])
    
    return q_result

# How to use this function    
#t_amb_set = [-20, -15, -7, 2, 7, 10, 12, 20]
#q_set = [4.89, 5.87, 7.60, 9.60, 11.40, 11.70, 12.20, 13.60]
#q = interpolateQ([11, 17.5], t_amb_set, q_set)

def heater_nominals(t_flow, val_35, val_55):
    """
        Compute the nominal output/consumption at the given flow temperature
        Inputs: (All arrays!)
            t_flow: Flow temperature in °C
            
            val_35: Q_35 or P_35 for all time steps
            
            val_55: Q_55 or P_55 for all time steps
    """
    val_out = np.zeros_like(t_flow)
    val_out[t_flow <= 35] = val_35[t_flow <= 35]
    val_out[t_flow >= 55] = val_55[t_flow >= 55]
    indexes = (t_flow > 35) * (t_flow < 55)
    val_out[indexes] = val_35[indexes] + (val_55[indexes] - val_35[indexes]) * (t_flow[indexes] - 35) / 20
    
    return val_out