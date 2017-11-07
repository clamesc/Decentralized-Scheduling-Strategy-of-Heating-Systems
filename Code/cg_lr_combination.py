# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 21:39:21 2014

@author: T_ohne_admin

        +-------------+
        |Initialize   | 
        +-----+-------+ 
              V
        +-----+-------+<________        __________+-----------------+
        |Solve Master |        |        |         |Solve Lagrangian |
        +-----+-------+        |        |         |Problem          |
              |                |        |         +------------+----+
              V            +---+----+   |                      |
    ____+-----+-------+    |Add New |   |               +------+----+
   |YES |Check Bound  |    |Columns |   |               |Update LR  |
   |    +-----+-------+    +---+----+   |               |Multipliers|
   |          |NO              |        |               +------+----+
   |          V                |        |                      |NO
   |    +-----+-------+        |________|___________+----------+----+
   |    |Solve Pricing|                 |        YES|Stop Inner Loop|
   |    +-----+-------+                 |           +----------+----+
   |          |                         |                      |
   |          V                         V                      |NO
   |    +-----+----------+_______+------+----+__________+------+----+
   |    |Check Optimality|NO     |Compute LR |          |Check Bound|
   |    +-----+----------+       |Bound      |          +------+----+
   |          |YES               +-----------+                 |YES
   |          V                                                |
   |_____>+---+----+___________________________________________|
          |LB found|
          +--------+

"""

from __future__ import division

import object_hp
import object_chp
import object_master
from constants import *
import numpy as np
import data
import h5py
import time
import os
from IPython.core.debugger import Tracer

def lr_cg_algorithm(day=0, t_init_hp=[], t_init_chp=[]):
    #============ Create Optimization model ============================#
    
    GlobVar.overall_count = 0
    GlobVar.inner_count = 0
    GlobVar.outer_count = 0
    GlobVar.final_count = 0
    GlobVar.first_iteration = True
    
    # Get data from input files
    Constants.t = day * 24 * 4
    dat = data.ModelData()

    # HP objects handle the HP pricing subproblems
    hp = []
    for i in range(len(dat.hps)):
	hp.append(object_hp.HP(dat.hps[i], dat.raw_temperature, i))
	if t_init_hp:
	    hp[-1].t_init = t_init_hp[i]

    # CHP objects handle the CHP pricing subproblems
    chp = []
    for i in range(len(dat.chps)):
	chp.append(object_chp.CHP(dat.chps[i], dat.raw_temperature, i))
	if t_init_chp:
	    chp[-1].t_init = t_init_chp[i]

    # Master object handles master problem
    mp = object_master.Master(len(chp), len(hp), dat.P_res)

    # Get initial feasible proposals
    for j in range(mp.number_hp):
	(mp.initial_costs_hp[j],
	mp.initial_proposals_hp[j]) = hp[j].compute_proposal(mp.initial_marginals)
    for k in range(mp.number_chp):
	(mp.initial_costs_chp[k],
	mp.initial_proposals_chp[k]) = chp[k].compute_proposal(mp.initial_marginals)

    start_time = time.clock()

    #============ Initial Column Generation ==================================================#

    while GlobVar.overall_count < Constants.cg_iterations:
      
	mp.solve_master()
	
	mp.master_time[GlobVar.outer_count] = time.clock() - start_time
	
	sum_obj_subs = 0
	for j in range(mp.number_hp):
	    (mp.pricing_costs_hp[GlobVar.overall_count, j],
	    mp.pricing_proposals_hp[GlobVar.overall_count, j]) = hp[j].compute_proposal(mp.marginals_mu_master[GlobVar.outer_count])
	    sum_obj_subs += hp[j].pricing_res_obj[GlobVar.overall_count]
	for k in range(mp.number_chp):
	    (mp.pricing_costs_chp[GlobVar.overall_count, k],
	    mp.pricing_proposals_chp[GlobVar.overall_count, k]) = chp[k].compute_proposal(mp.marginals_mu_master[GlobVar.outer_count])
	    sum_obj_subs += chp[k].pricing_res_obj[GlobVar.overall_count]

	mp.update_lr_bound(sum_obj_subs)
	
	# Get computation time
	mp.sub_time[GlobVar.overall_count] = time.clock() - start_time
	
	mp.update_plot_cg()
	
	GlobVar.outer_count += 1
	GlobVar.overall_count += 1

    #============ Outer Loop =================================================================#
    break_outer_loop = False
    break_inner_loop = False
    while GlobVar.outer_count < Constants.outer_max:
	
	#============ Inner Loop ======================================================#
	GlobVar.inner_count = 0
	while GlobVar.inner_count < Constants.inner_loops[GlobVar.outer_count-Constants.cg_iterations]:
	    
	    # Compute Sugradient and Stepsize
	    subgradient = mp.compute_subgradient()
	    alpha = Constants.alpha
	    stepsize = alpha * (mp.lin_obj_values[GlobVar.outer_count-1] - mp.lr_bound) / np.sum(np.square(subgradient))
	    
	    # Update Langrange Multipliers (Shadowprices)
	    for t in range(Constants.timesteps):
		if GlobVar.first_iteration:
		    marginal_mu_new = mp.marginals_mu_master[Constants.cg_iterations-1, t] + stepsize * subgradient[t]
		else:
		    marginal_mu_new = mp.marginals_mu_subgradient[GlobVar.overall_count-Constants.cg_iterations-1, t] + stepsize * subgradient[t]
		if marginal_mu_new <= Constants.r_el * Constants.dt:
		    mp.marginals_mu_subgradient[GlobVar.overall_count-Constants.cg_iterations, t] = Constants.r_el * Constants.dt
		elif marginal_mu_new >= Constants.k_el * Constants.dt:
		    mp.marginals_mu_subgradient[GlobVar.overall_count-Constants.cg_iterations, t] = Constants.k_el * Constants.dt
		else:
		    mp.marginals_mu_subgradient[GlobVar.overall_count-Constants.cg_iterations, t] = marginal_mu_new
	    GlobVar.first_iteration = False
	    # Solve Pricing Problems
	    sum_obj_subs = 0
	    for j in range(mp.number_hp):
		(mp.pricing_costs_hp[GlobVar.overall_count, j],
		mp.pricing_proposals_hp[GlobVar.overall_count, j]) = hp[j].compute_proposal(mp.marginals_mu_subgradient[GlobVar.overall_count-Constants.cg_iterations])
		sum_obj_subs += hp[j].pricing_res_obj[GlobVar.overall_count]
	    for k in range(mp.number_chp):
		(mp.pricing_costs_chp[GlobVar.overall_count, k],
		mp.pricing_proposals_chp[GlobVar.overall_count, k]) = chp[k].compute_proposal(mp.marginals_mu_subgradient[GlobVar.overall_count-Constants.cg_iterations])
		sum_obj_subs += chp[k].pricing_res_obj[GlobVar.overall_count]
	    
	    # Check Lagrangian Bound
	    mp.update_lr_bound(sum_obj_subs)
	    
	    # Get computation time
	    mp.sub_time[GlobVar.overall_count] = time.clock() - start_time
	    
	    GlobVar.overall_count += 1
	    
	    if (mp.lin_obj_values[GlobVar.outer_count-1] - mp.lr_bound)/mp.lin_obj_values[GlobVar.outer_count-1] < Constants.eps or (time.clock() - start_time) > Constants.pricing_time_limit:
		break_inner_loop = True
		break
	    
	    GlobVar.inner_count += 1

	#============ End Inner Loop ==================================================#
	
	if break_inner_loop:
	    break
	
	if GlobVar.overall_count == Constants.random_proposals_in_master:
	    GlobVar.random_proposals_in_master = False
	
	# Solve Master
	mp.solve_master()
	
	mp.master_time[GlobVar.outer_count] = time.clock() - start_time
	
	mp.update_plot_lr()
	
	if (mp.lin_obj_values[GlobVar.outer_count] - mp.lr_bound)/mp.lin_obj_values[GlobVar.outer_count] < Constants.eps or (time.clock() - start_time) > Constants.pricing_time_limit:
	    break_outer_loop = True
	    break

	GlobVar.outer_count += 1
	
    #============ End Outer Loop =============================================================#

    if break_outer_loop:
	mp.final_marginals = mp.marginals_mu_subgradient[GlobVar.overall_count-Constants.cg_iterations-1]
    elif break_inner_loop:
	mp.final_marginals = mp.marginals_mu_subgradient[GlobVar.overall_count-Constants.cg_iterations-1]
	mp.update_plot_inner_break()
    else:
	mp.final_marginals = mp.marginals_mu_subgradient[-1]
	
    while GlobVar.final_count < Constants.final_max:
	
	for j in range(mp.number_hp):
	    (mp.final_costs_hp[GlobVar.final_count, j],
	    mp.final_proposals_hp[GlobVar.final_count, j]) = hp[j].compute_proposal(mp.final_marginals, True)
	for k in range(mp.number_chp):
	    (mp.final_costs_chp[GlobVar.final_count, k],
	    mp.final_proposals_chp[GlobVar.final_count, k]) = chp[k].compute_proposal(mp.final_marginals, True)

	#mp.solve_master(True)
	
	mp.update_plot_final(GlobVar.overall_count)
	
	mp.final_time[GlobVar.final_count] = time.clock() - start_time
	
	GlobVar.final_count += 1

    GlobVar.final_count -= 1
    mp.solve_master(True)

    f = h5py.File(Constants.path + "/lr_" + str(day) + ".hdf5", "w")

    for i in range(len(hp)):
	g = f.create_group("HP_"+str(i))
	g.create_dataset("x_p", data = hp[i].pricing_res_x)
	g.create_dataset("y_p", data = hp[i].pricing_res_y)
	g.create_dataset("T_p", data = hp[i].pricing_res_T)
	g.create_dataset("P_p", data = hp[i].pricing_res_P)
	g.create_dataset("z_p", data = hp[i].pricing_res_obj)
	g.create_dataset("x_f", data = hp[i].final_res_x)
	g.create_dataset("y_f", data = hp[i].final_res_y)
	g.create_dataset("T_f", data = hp[i].final_res_T)
	g.create_dataset("P_f", data = hp[i].final_res_P)
	g.create_dataset("z_f", data = hp[i].final_res_obj)
    for j in range(len(chp)):
	g = f.create_group("CHP_"+str(j))
	g.create_dataset("x_p", data = chp[j].pricing_res_x)
	g.create_dataset("y_p", data = chp[j].pricing_res_y)
	g.create_dataset("T_p", data = chp[j].pricing_res_T)
	g.create_dataset("P_p", data = chp[j].pricing_res_P)
	g.create_dataset("Q_p", data = chp[j].pricing_res_Q)
	g.create_dataset("z_p", data = chp[j].pricing_res_obj)
	g.create_dataset("c_p", data = chp[j].pricing_res_costs)
	g.create_dataset("x_f", data = chp[j].final_res_x)
	g.create_dataset("y_f", data = chp[j].final_res_y)
	g.create_dataset("T_f", data = chp[j].final_res_T)
	g.create_dataset("P_f", data = chp[j].final_res_P)
	g.create_dataset("Q_f", data = chp[j].final_res_Q)
	g.create_dataset("z_f", data = chp[j].final_res_obj)
	g.create_dataset("c_f", data = chp[j].final_res_costs)
	
    f.create_dataset("pricing_costs_chp", data = mp.pricing_costs_chp)
    f.create_dataset("pricing_costs_hp", data = mp.pricing_costs_hp)
    f.create_dataset("pricing_proposals_chp", data = mp.pricing_proposals_chp)
    f.create_dataset("pricing_proposals_hp", data = mp.pricing_proposals_hp)

    f.create_dataset("final_costs_chp", data = mp.final_costs_chp)
    f.create_dataset("final_costs_hp", data = mp.final_costs_hp)
    f.create_dataset("final_proposals_chp", data = mp.final_proposals_chp)
    f.create_dataset("final_proposals_hp", data = mp.final_proposals_hp)

    f.create_dataset("initial_costs_chp", data = mp.initial_costs_chp)
    f.create_dataset("initial_costs_hp", data = mp.initial_costs_hp)
    f.create_dataset("initial_proposals_chp", data = mp.initial_proposals_chp)
    f.create_dataset("initial_proposals_hp", data = mp.initial_proposals_hp)

    f.create_dataset("marginals_sigma_chp", data = mp.marginals_sigma_chp)
    f.create_dataset("marginals_sigma_hp", data = mp.marginals_sigma_hp)
    f.create_dataset("marginals_mu_master", data = mp.marginals_mu_master)
    f.create_dataset("marginals_mu_subgradient", data = mp.marginals_mu_subgradient)
    f.create_dataset("initial_marginals", data = mp.initial_marginals)
    f.create_dataset("final_marginals", data = mp.final_marginals)

    f.create_dataset("lin_obj_values", data = mp.lin_obj_values)
    f.create_dataset("int_obj_values", data = mp.int_obj_values)

    f.create_dataset("res_lr_bounds_master", data = mp.res_lr_bounds_master)
    f.create_dataset("res_lr_bounds_sugradient", data = mp.res_lr_bounds_subgradient)

    f.create_dataset("sub_time", data = mp.sub_time)
    f.create_dataset("master_time", data = mp.master_time)
    f.create_dataset("final_time", data = mp.final_time)
    
    f.create_dataset("final_hp", data = mp.final_hp)
    f.create_dataset("final_chp", data = mp.final_chp)

    f.attrs["max_time_subs"] = Constants.max_time_subs
    f.attrs["max_time_master"] = Constants.max_time_master
    f.attrs["timesteps"] = Constants.timesteps
    f.attrs["t"] = Constants.t
    f.attrs["cg_iterations"] = Constants.cg_iterations
    f.attrs["inner_loops"] = Constants.inner_loops
    f.attrs["final_max"] = Constants.final_max
    f.attrs["outer_loops"] = Constants.outer_loops
    f.attrs["outer_max"] = Constants.outer_max
    f.attrs["overall_max"] = Constants.overall_max
    f.attrs["subgradient_max"] = Constants.subgradient_max
    f.attrs["eps"] = Constants.eps
    f.attrs["alpha"] = Constants.alpha
    f.attrs["t_bivalent"] = Constants.t_bivalent
    f.attrs["pricing_MIPGap"] = Constants.pricing_MIPGap
    f.attrs["final_MIPGap"] = Constants.final_MIPGap
    f.attrs["initial_in_master"] = Constants.initial_in_master
    f.attrs["pricing_time_limit"] = Constants.pricing_time_limit

    f.close()
    
    return [hp[j].final_res_T[mp.final_hp[j],95] for j in range(mp.number_hp)], [chp[j].final_res_T[mp.final_chp[j],95] for j in range(mp.number_chp)]
  
d=55
f = h5py.File("/home/qwertzuiopu/Data/lr_"+str(d-1)+".hdf5")
array = f["final_hp"]
fhp = np.empty(array.shape)
array.read_direct(fhp)
array = f["final_chp"]
fchp = np.empty(array.shape)
array.read_direct(fchp)
t_hp = []
t_chp = []
for i in range(51):
    t_hp.append(f["HP_"+str(i)+"/T_f"][fhp[i], 95])
    t_chp.append(f["CHP_"+str(i)+"/T_f"][fchp[i], 95])
while d < 56:
    t_hp, t_chp = lr_cg_algorithm(d,t_hp,t_chp)
    d += 1