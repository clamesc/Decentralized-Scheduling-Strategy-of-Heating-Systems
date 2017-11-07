# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 09:08:31 2014

@author: T_ohne_admin
"""

from constants import *
import numpy as np
import basic_functions.heating_tech
import model_subproblem_hp
import heat_pumps.HP_characteristics
import basic_functions.read_txt as rtxt
import os

class HP(Storage):
    """
    Overview of all methods and usage of this class
    """
    
    def __init__(self, data, t_ambient, i):
        """
        """        
        self.nr = i
        self.sto_m = data.m_sto
        self.sto_UA = data.U_sto
        self.t_init = data.T_init
        self.t_max = data.T_max
        self.Q_sec_nom = data.Q_sec
        self.eta_sec = data.eta_sec
        self.t_outside = t_ambient
        self.heatdemand = self.compute_heatdemand(data.Q_dem)
        self.t_flow = self.compute_t_flow(t_ambient)
        self.P_dem = data.P_dem
        
        self.Q_prim, self.P_prim = heat_pumps.HP_characteristics.get_hp_data(data.Q_nom, t_ambient, self.t_flow)
        
        self.pricing_res_x = np.zeros([self.overall_max, self.timesteps], dtype="bool_")
        self.pricing_res_y = np.zeros([self.overall_max, self.timesteps], dtype="bool_")
        self.pricing_res_T = np.zeros([self.overall_max, self.timesteps])
        self.pricing_res_P = np.zeros([self.overall_max, self.timesteps])
        self.pricing_res_obj = np.zeros([self.overall_max])
        
        self.final_res_x = np.zeros([self.final_max, self.timesteps], dtype="bool_")
        self.final_res_y = np.zeros([self.final_max, self.timesteps], dtype="bool_")
        self.final_res_T = np.zeros([self.final_max, self.timesteps])
        self.final_res_P = np.zeros([self.final_max, self.timesteps])
        self.final_res_obj = np.zeros([self.final_max])
        
        
    def compute_t_flow(self, t_ambient):
        """
        Compute the flow temperature based on the ambient temperature (in °C)
        """
        return basic_functions.heating_tech.heatingCurve(t_ambient+273.15) - 273.15
        
    def compute_heatdemand(self, heatdemand):
        """
        For testing purposes, this has not really been implemented yet.
        In future works, a heat demand forecast is preferable - now the given
        heat demand is accepted as the house's heat demand.
        """
        return heatdemand
    
    def compute_el_demand(self, raw_slp, users):
	"""
	Compute the electricity consumption based on the standard load profile
      
	    Parameters:
                
                slp --> original standard load profile (Array)
                
                users --> inhabitant living in this apartment / building
            
                timestep_length --> length of one time step in seconds
            
            Result:
            
                Electricity consumption in Watt (Array)
        """
        average_demand = 872.6 * users + 1497.1
        scaling_factor = average_demand * 3600 / Constants.dt
        return raw_slp * scaling_factor
    
    def compute_proposal(self, marginals, final=False):
        """
        This function computes a new proposal (P and k).
        Internally, the results of the subproblem are stored.
        If this is the first time in the current optimization period that new
            proposals have to generated, _iteration_ has to be set to 0
        """
        if final:
	    (self.final_res_obj[GlobVar.final_count],
	     self.final_res_x[GlobVar.final_count],
	     self.final_res_y[GlobVar.final_count],
	     self.final_res_T[GlobVar.final_count],
	     self.final_res_P[GlobVar.final_count]) = model_subproblem_hp.optimize(self, marginals, final)
	    return 0, self.final_res_P[GlobVar.final_count]
	else:
	    (self.pricing_res_obj[GlobVar.overall_count],
	     self.pricing_res_x[GlobVar.overall_count],
	     self.pricing_res_y[GlobVar.overall_count],
	     self.pricing_res_T[GlobVar.overall_count],
	     self.pricing_res_P[GlobVar.overall_count]) = model_subproblem_hp.optimize(self, marginals, final)
	    return 0, self.pricing_res_P[GlobVar.overall_count]
        
    def get_optimal_schedule(self, lambdas):
        """
        This function returns the scheduling of the primary and secondary
        heating device as well as the resulting storage temperature:
        x, y, T, P
        """
        index = np.argmax(lambdas)
        return (self.res_x[index], self.res_y[index], self.res_T[index], self.res_P[index])