# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 21:20:59 2014

@author: T_ohne_admin
"""
import constants
import os
import numpy as np
import basic_functions.read_txt as rtxt
import microgrid_functions.renewables as renewables

class ModelData(constants.Constants):
  
    class hp(object):
	def __init__(self, hp_nom, 
			   hp_sec, 
			   hp_sec_eta, 
			   hp_sto_m, 
			   hp_sto_U, 
			   hp_t_init, 
			   hp_t_max, 
			   hp_p_dem, 
			   hp_q_dem):
	    
	    self.Q_nom = hp_nom
	    self.Q_sec = hp_sec
	    self.eta_sec = hp_sec_eta
	    self.m_sto = hp_sto_m
	    self.U_sto = hp_sto_U
	    self.T_init = hp_t_init
	    self.T_max = hp_t_max
	    self.P_dem = hp_p_dem
	    self.Q_dem = hp_q_dem
	    
    class chp(object):
	def __init__(self, chp_nom, 
			   chp_eta, 
			   chp_sigma, 
			   chp_sec, 
			   chp_sec_eta, 
			   chp_sto_m, 
			   chp_sto_U, 
			   chp_t_init, 
			   chp_t_max, 
			   chp_p_dem, 
			   chp_q_dem):
	    
	    self.Q_nom = chp_nom
	    self.eta = chp_eta
	    self.sigma = chp_sigma
	    self.Q_sec = chp_sec
	    self.eta_sec = chp_sec_eta
	    self.m_sto = chp_sto_m
	    self.U_sto = chp_sto_U
	    self.T_init = chp_t_init
	    self.T_max = chp_t_max
	    self.P_dem = chp_p_dem
	    self.Q_dem = chp_q_dem
    
    def __init__(self):
	hp_q_dem  = 		np.transpose(rtxt.read_multiple_indexed_file(os.getcwd() + "/input_data/heat_demand_hp.csv",  delimiter = ";")[self.t:self.t+self.timesteps,:])
	hp_p_dem  = 		np.transpose(rtxt.read_multiple_indexed_file(os.getcwd() + "/input_data/el_demand_hp.csv",  delimiter = ";")[self.t:self.t+self.timesteps,:])
	chp_q_dem = 		np.transpose(rtxt.read_multiple_indexed_file(os.getcwd() + "/input_data/heat_demand_chp.csv", delimiter = ";")[self.t:self.t+self.timesteps,:])
	chp_p_dem  = 		np.transpose(rtxt.read_multiple_indexed_file(os.getcwd() + "/input_data/el_demand_chp.csv",  delimiter = ";")[self.t:self.t+self.timesteps,:])
	hp_input_data = 	np.transpose(rtxt.read_multiple_indexed_file(os.getcwd() + "/input_data/input_data_hp.csv", delimiter = ";"))
	chp_input_data = 	np.transpose(rtxt.read_multiple_indexed_file(os.getcwd() + "/input_data/input_data_chp.csv", delimiter = ";"))
	#self.raw_slp = 		rtxt.read_single_indexed_file(os.getcwd() + "/input_data/standard_load_profile.txt")[self.t:self.t+self.timesteps]
	self.raw_temperature = 	rtxt.read_single_indexed_file(os.getcwd()  + "/input_data/temperature.txt")[self.t:self.t+self.timesteps]
	
	self.hps = []
	self.chps = []
	for i in range(len(hp_input_data)):
	    self.hps.append(self.hp(hp_input_data[i,0], 
				    hp_input_data[i,1], 
				    hp_input_data[i,2], 
				    hp_input_data[i,3], 
				    hp_input_data[i,4], 
				    hp_input_data[i,6], 
				    hp_input_data[i,5], 
				    hp_p_dem[i], 
				    hp_q_dem[i]))
	
	for i in range(len(chp_input_data)):
	    self.chps.append(self.chp(chp_input_data[i,0], 
				      chp_input_data[i,2], 
				      chp_input_data[i,1], 
				      chp_input_data[i,3], 
				      1, 
				      chp_input_data[i,4], 
				      chp_input_data[i,5], 
				      chp_input_data[i,7], 
				      chp_input_data[i,6], 
				      chp_p_dem[i], 
				      chp_q_dem[i]))
	
	self.P_res = 		renewables.compute_renewables(rtxt.read_single_indexed_file(os.getcwd()   + "/input_data/wind_speed.txt")[self.t:self.t+self.timesteps],
							      rtxt.read_single_indexed_file(os.getcwd()   + "/input_data/sun_direct.txt")[self.t:self.t+self.timesteps],
							      len(self.hps), len(self.chps))
	