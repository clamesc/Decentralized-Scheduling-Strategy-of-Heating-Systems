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
import pdb
from IPython.core.debugger import Tracer

def lr_cg_algorithm(day=0, t_init_hp=[], t_init_chp=[]):
    #============ Create Optimization model ============================#
    
    GlobVar.overall_count = 0
    GlobVar.inner_count = 0
    GlobVar.outer_count = 0
    GlobVar.final_count = 0
    GlobVar.first_iteration = True
    Constants.plot = False
    Constants.cg_iterations = 50
    Constants.inner_loops = []
    Constants.final_max = 0
    Constants.outer_loops = 0    
    Constants.outer_max = Constants.cg_iterations
    Constants.overall_max = Constants.cg_iterations
    Constants.subgradient_max = 0
    
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
	mp.initial_proposals_hp[j]) = hp[j].compute_proposal(np.zeros([Constants.timesteps]))
    for k in range(mp.number_chp):
	(mp.initial_costs_chp[k],
	mp.initial_proposals_chp[k]) = chp[k].compute_proposal(np.zeros([Constants.timesteps]))

    start_time = time.clock()

    #============ Initial Column Generation ==================================================#

    while GlobVar.overall_count < Constants.cg_iterations and (time.clock() - start_time) < 600:
      
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
    
    f = h5py.File(Constants.path + "/cg_" + str(day) + ".hdf5", "w")

    for i in range(len(hp)):
	g = f.create_group("HP_"+str(i))
	g.create_dataset("x_p", data = hp[i].pricing_res_x)
	g.create_dataset("y_p", data = hp[i].pricing_res_y)
	g.create_dataset("T_p", data = hp[i].pricing_res_T)
	g.create_dataset("P_p", data = hp[i].pricing_res_P)
	g.create_dataset("z_p", data = hp[i].pricing_res_obj)
    for j in range(len(chp)):
	g = f.create_group("CHP_"+str(j))
	g.create_dataset("x_p", data = chp[j].pricing_res_x)
	g.create_dataset("y_p", data = chp[j].pricing_res_y)
	g.create_dataset("T_p", data = chp[j].pricing_res_T)
	g.create_dataset("P_p", data = chp[j].pricing_res_P)
	g.create_dataset("Q_p", data = chp[j].pricing_res_Q)
	g.create_dataset("z_p", data = chp[j].pricing_res_obj)
	g.create_dataset("c_p", data = chp[j].pricing_res_costs)
	
    f.create_dataset("pricing_costs_chp", data = mp.pricing_costs_chp)
    f.create_dataset("pricing_costs_hp", data = mp.pricing_costs_hp)
    f.create_dataset("pricing_proposals_chp", data = mp.pricing_proposals_chp)
    f.create_dataset("pricing_proposals_hp", data = mp.pricing_proposals_hp)

    f.create_dataset("marginals_sigma_chp", data = mp.marginals_sigma_chp)
    f.create_dataset("marginals_sigma_hp", data = mp.marginals_sigma_hp)
    f.create_dataset("marginals_mu_master", data = mp.marginals_mu_master)

    f.create_dataset("lin_obj_values", data = mp.lin_obj_values)

    f.create_dataset("res_lr_bounds_master", data = mp.res_lr_bounds_master)

    f.create_dataset("sub_time", data = mp.sub_time)
    f.create_dataset("master_time", data = mp.master_time)

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
  
#lr_cg_algorithm(0)
d=85
while d < 365:
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
    lr_cg_algorithm(d, t_hp, t_chp)
    d += 1