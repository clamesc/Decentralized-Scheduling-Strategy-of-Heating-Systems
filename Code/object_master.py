# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 21:45:25 2014

@author: T_ohne_admin
"""


# http://en.wikipedia.org/wiki/Branch_and_price

from constants import *
import numpy as np
import model_masterproblem
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Master(Constants):
    """
    This is the Master class that handles all attributes relevant to the masterproblem
    """
    
    def __init__(self, number_chp, number_hp, P_res):
        """
        Initialization parameters are:
            number_chp: Number of installed CHP units
            
            number_hp:  Number of installed HP units
        """
        
        self.number_chp = number_chp
        self.number_hp  = number_hp
        self.P_renewables = P_res
        
        self.pricing_costs_chp = np.zeros([self.overall_max, number_chp])
        self.pricing_costs_hp  = np.zeros([self.overall_max, number_hp])
        self.pricing_proposals_chp = np.zeros([self.overall_max, number_chp, self.timesteps])
        self.pricing_proposals_hp  = np.zeros([self.overall_max, number_hp, self.timesteps])
        
        self.final_costs_chp = np.zeros([self.final_max, number_chp])
        self.final_costs_hp  = np.zeros([self.final_max, number_hp])
        self.final_proposals_chp = np.zeros([self.final_max, number_chp, self.timesteps])
        self.final_proposals_hp  = np.zeros([self.final_max, number_hp, self.timesteps])
        
        self.initial_costs_chp = np.zeros([number_chp])
        self.initial_costs_hp  = np.zeros([number_hp])
        self.initial_proposals_chp = np.zeros([number_chp, self.timesteps])
        self.initial_proposals_hp  = np.zeros([number_hp, self.timesteps])
        
        self.initial_marginals = np.random.uniform(Constants.r_el*Constants.dt, Constants.k_el*Constants.dt, Constants.timesteps)
        
        self.final_marginals = np.zeros([self.timesteps])
        
        self.marginals_sigma_chp = np.zeros([self.outer_max, number_chp])
        self.marginals_sigma_hp  = np.zeros([self.outer_max, number_hp])
        self.marginals_mu_master = np.zeros([self.outer_max, self.timesteps])
        self.marginals_mu_subgradient = np.zeros([self.subgradient_max, self.timesteps])
        
        self.lin_obj_values = np.zeros([self.outer_max])
        
        self.int_obj_values = np.zeros([self.final_max])
        
        self.lr_bound = 0
        self.res_lr_bounds_master = np.zeros([self.outer_max])
        self.res_lr_bounds_subgradient = np.zeros([self.subgradient_max])
        
        self.sub_time = np.zeros([Constants.overall_max])
        self.master_time = np.zeros([Constants.outer_max])
        self.final_time = np.zeros([Constants.final_max])
        
        self.initialize_plot()
        
        self.final_hp = np.zeros([number_hp])
        self.final_chp = np.zeros([number_chp])
        
        
    def solve_master(self, final = False):
        """
        Compute new marginals, based on proposed costs and electricity proposals
        
        In case of initializing Branch&Price with the masterproblem, 
            use empty lists as proposals:
        update_proposals([], [], [], [])
        
        This function returns the objective value of the masterproblem as 
            well as the marginals of the resource constraint (electricity balance)
        """
        
        if final:
	    (self.int_obj_values[GlobVar.final_count],
	     self.final_hp,
	     self.final_chp) = model_masterproblem.optimize(self, final)
	else:
	    (self.lin_obj_values[GlobVar.outer_count],
	     self.marginals_mu_master[GlobVar.outer_count],
	     self.marginals_sigma_chp[GlobVar.outer_count],
	     self.marginals_sigma_hp[GlobVar.outer_count]) = model_masterproblem.optimize(self, final)
	
    def update_lr_bound(self, sum_obj_subs):
	if GlobVar.overall_count < Constants.cg_iterations:
	    new_lr_bound = sum_obj_subs - np.sum(self.P_renewables * self.marginals_mu_master[GlobVar.outer_count])
	    self.lr_bound = max(self.lr_bound, new_lr_bound)
	    self.res_lr_bounds_master[GlobVar.outer_count] = new_lr_bound
	else:
	    new_lr_bound = sum_obj_subs - np.sum(self.P_renewables * self.marginals_mu_subgradient[GlobVar.overall_count-Constants.cg_iterations])
	    print new_lr_bound
	    self.lr_bound = max(self.lr_bound, new_lr_bound)
	    self.res_lr_bounds_subgradient[GlobVar.overall_count-Constants.cg_iterations] = new_lr_bound
    
    def compute_subgradient(self):
	if GlobVar.first_iteration:
	    return np.fromiter(( - np.sum(self.pricing_proposals_hp[Constants.cg_iterations-1, :, t])
				 - np.sum(self.pricing_proposals_chp[Constants.cg_iterations-1, :, t]) 
				 - self.P_renewables[t] 
				 for t in range(self.timesteps)), np.float)
	else:
	    return np.fromiter(( - np.sum(self.pricing_proposals_hp[GlobVar.overall_count-1, :, t])
				 - np.sum(self.pricing_proposals_chp[GlobVar.overall_count-1, :, t])
				 - self.P_renewables[t]
				 for t in range(self.timesteps)), np.float)
    
    def finalize(self):
        """
        Finalize the Branch&Price process by solving the masterproblem with 
            binaries instead of continuous variables
        """
        return model_masterproblem.optimize(self, True)
    
    def initialize_plot(self):
	if Constants.plot:
	    plt.plot(0, '-')
	    plt.ion()
	    plt.show()
	    self.obj_it = np.arange(Constants.cg_iterations)
    
    def update_plot_cg(self):
	if Constants.plot:
	    plt.plot(range(1, GlobVar.overall_count+2), self.lin_obj_values[:GlobVar.outer_count+1], 'k-', range(1, GlobVar.overall_count+2), self.res_lr_bounds_master[:GlobVar.outer_count+1], 'b-')
	    plt.title("Column generation")
	    plt.xlabel("Number of iterations")
	    plt.ylabel("Costs in Euro")
	    plt.ylim([min(self.res_lr_bounds_master[:GlobVar.outer_count+1])-250,max(self.lin_obj_values[:GlobVar.outer_count+1])+250])
	    plt.grid(True)
	    plt.draw()
	
    def show_plot(self):
	if Constants.plot:
	    plt.ioff()
	    plt.show()
	
    def update_plot_lr(self):
	if Constants.plot:
	    loops = sum(Constants.inner_loops[:(GlobVar.outer_count - Constants.cg_iterations+1)])
	    self.obj_it = np.append(self.obj_it, Constants.cg_iterations-1 + loops)
	    plt.plot([x+1 for x in self.obj_it], 	self.lin_obj_values[:GlobVar.outer_count+1], 							'k.-',
		      np.arange(Constants.cg_iterations, Constants.cg_iterations+loops+1), 	np.append(self.res_lr_bounds_master[Constants.cg_iterations-1], self.res_lr_bounds_subgradient[:loops]), 	'r-')
	    plt.title("Column generation with subgradient optimization")
	    plt.xlabel("Number of iterations")
	    plt.ylabel("Costs in Euro")
	    plt.ylim([self.res_lr_bounds_master[Constants.cg_iterations-1]-250,max(self.lin_obj_values[:GlobVar.outer_count+1])+250])
	    plt.grid(True)
	    plt.draw()

    def update_plot_inner_break(self):
	if Constants.plot:
	    plt.plot(range(GlobVar.overall_count-1-GlobVar.inner_count,GlobVar.overall_count+1), self.res_lr_bounds_subgradient[GlobVar.overall_count-2-GlobVar.inner_count-Constants.cg_iterations:GlobVar.overall_count-Constants.cg_iterations], 'r-')
	    #plt.axhline(y=self.lin_obj_values[GlobVar.outer_count-1], xmin=0, xmax=1, linewidth=1, color = 'k')
	    plt.draw()

    def update_plot_final(self, pricing_end):
	if Constants.plot:
	    plt.plot(range(pricing_end+1, pricing_end+2+GlobVar.final_count), self.int_obj_values[:GlobVar.final_count+1], 'ko')
	    plt.ylim([self.res_lr_bounds_master[0]-250,max(self.lin_obj_values[0], self.int_obj_values[0])+250])
	    plt.draw()