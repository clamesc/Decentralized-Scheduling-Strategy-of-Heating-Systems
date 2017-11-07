# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 22:00:29 2014

@author: T_ohne_admin
"""

from gurobipy import *
import numpy as np
from constants import *
import microgrid_functions.renewables as renewables
import os
import basic_functions.read_txt as rtxt

def optimize(mp, final_iteration=False):
    """
    
    """
    timesteps = mp.timesteps
    max_time=Constants.max_time_master

    chp_number = mp.number_chp
    hp_number  = mp.number_hp
    
    if final_iteration:
	chp_number_props = GlobVar.final_count +1
	prop_chp = mp.final_proposals_chp[:chp_number_props]
	k_chp = mp.final_costs_chp[:chp_number_props]
    else:
	if GlobVar.overall_count == 0:
	    chp_number_props = GlobVar.overall_count+1
	    prop_chp = np.array([mp.initial_proposals_chp])
	    k_chp = np.array([mp.initial_costs_chp])
	else:
	    chp_number_props = GlobVar.overall_count
	    prop_chp = mp.pricing_proposals_chp[:chp_number_props]
	    k_chp = mp.pricing_costs_chp[:chp_number_props]
	    if GlobVar.random_proposals_in_master:
		prop_chp = np.append(prop_chp, [mp.initial_proposals_chp], axis=0)
		k_chp = np.append(k_chp, [mp.initial_costs_chp], axis=0)
		chp_number_props += 1
    
    if final_iteration:
	hp_number_props = GlobVar.final_count + 1
	prop_hp  = mp.final_proposals_hp[:hp_number_props]
    else:
	if GlobVar.overall_count == 0:
	    hp_number_props = GlobVar.overall_count+1
	    prop_hp  = np.array([mp.initial_proposals_hp])
	else:
	    hp_number_props = GlobVar.overall_count
	    prop_hp  = mp.pricing_proposals_hp[:hp_number_props]
	    if GlobVar.random_proposals_in_master:
		prop_hp  = np.append(prop_hp, [mp.initial_proposals_hp], axis=0)
		hp_number_props += 1

    '''
    if chp_number_props == 0:
        bounds_chp = np.ones((1, chp_number))
        bounds_hp = np.ones((1, hp_number))
    else:
        if mp.count_iteration > 2:
            bounds_chp = np.array(mp.bounds_chp)
            bounds_hp  = np.array(mp.bounds_hp)
        else:
            bounds_chp = np.ones((1, chp_number))
            bounds_hp = np.ones((1, hp_number))
    '''
    
    P_ren = mp.P_renewables
        
    dt = mp.dt
    k_el = mp.k_el
    r_el = mp.r_el
    
    # Gurobi optimization model
    try:
        # Create a new model
        model = Model("masterproblem")
        model.Params.OutputFlag = 0

        # Create variables with one or more sets:
        l_chp = {} # Weighting variables of the CHP proposals
        l_hp  = {} # Weighting variables of the HP proposals
        
        P_imp = {} # Imported electricity
        P_exp = {} # Exported electricity
   
        if final_iteration:
            for p in xrange(hp_number_props):
                for j in xrange(hp_number):
                    l_hp[p,j] = model.addVar(vtype=GRB.BINARY, name="l_hp_"+str(p)+"_"+str(j))
                    
            for p in xrange(chp_number_props):
                for k in xrange(chp_number):
                    l_chp[p,k] = model.addVar(vtype=GRB.BINARY, name="l_chp_"+str(p)+"_"+str(k))
        else:
            for p in xrange(hp_number_props):
                for j in xrange(hp_number):
                    l_hp[p,j] = model.addVar(vtype=GRB.CONTINUOUS, name="l_hp_"+str(p)+"_"+str(j), lb=0.0, ub=1.0)
                    
            for p in xrange(chp_number_props):
                for k in xrange(chp_number):
                    l_chp[p,k] = model.addVar(vtype=GRB.CONTINUOUS, name="l_chp_"+str(p)+"_"+str(k), lb=0.0, ub=1.0)
	
        for t in xrange(timesteps):
            P_imp[t] = model.addVar(vtype=GRB.CONTINUOUS, name="P_imp_"+str(t), lb=0.0)
            P_exp[t] = model.addVar(vtype=GRB.CONTINUOUS, name="P_exp_"+str(t), lb=0.0)

        # Integrate new variables into the model
        model.update()    
	
        # Set objective
        costs_electricity = quicksum(P_imp[t]*k_el - P_exp[t]*r_el for t in range(timesteps)) *dt
        costs_gas_chp = quicksum(quicksum(k_chp[p,k] * l_chp[p,k] for p in range(chp_number_props)) for k in range(chp_number))     
        model.setObjective(costs_electricity + costs_gas_chp, GRB.MINIMIZE)
        
        # Add constraints
        # Electricity balance:
        for t in xrange(timesteps):
            hp_proposal  = quicksum(quicksum(prop_hp[p,j,t]  * l_hp[p,j]  for p in range(hp_number_props))  for j in range(hp_number))
            chp_proposal = quicksum(quicksum(prop_chp[p,k,t] * l_chp[p,k] for p in range(chp_number_props)) for k in range(chp_number))
            model.addConstr(hp_proposal + chp_proposal + P_ren[t] + P_imp[t] - P_exp[t] ==  0 , "ElectricityBalance_"+str(t))
        
        # Convexity constraints
        for j in xrange(hp_number):
            model.addConstr(quicksum(l_hp[p,j] for p in range(hp_number_props)) == 1, "Convex_hp_"+str(j))
        for k in xrange(chp_number):
            model.addConstr(quicksum(l_chp[p,k] for p in range(chp_number_props)) == 1, "Convex_chp_"+str(k))
        
        # Set Gurobi parameters
        model.Params.Presolve = 0
#        model.Params.MIPGap = 0.01
	if not final_iteration:
	    model.Params.TimeLimit = max_time
	else:
	    model.Params.TimeLimit = 15
        
        # Run model
        model.optimize()
        
        # Print final solution
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            r_obj = model.ObjVal
            print("Current objective of the master problem: " + str(r_obj))
            
            if final_iteration:
                r_hp = np.zeros((hp_number))
                r_chp = np.zeros((chp_number))
                
                for p in xrange(hp_number_props):
                    for j in xrange(hp_number):
                        if round(l_hp[p,j].X)==1:
			    r_hp[j] = p
                for p in xrange(chp_number_props):
                    for k in xrange(chp_number):
                        if round(l_chp[p,k].X)==1:
			    r_chp[k] = p
            else:
                r_sigma_hp  = np.zeros(hp_number)
                r_sigma_chp = np.zeros(chp_number)
                r_pi = np.zeros(timesteps)
                            
                for t in xrange(timesteps):
                    r_pi[t] = (model.getConstrByName("ElectricityBalance_"+str(t))).Pi #.getAttr("Pi")
    
                for j in xrange(hp_number):
                    r_sigma_hp[j]  = (model.getConstrByName("Convex_hp_"+str(j))).Pi #.getAttr("Pi")
                    
                for k in xrange(chp_number):
                    r_sigma_chp[k] = (model.getConstrByName("Convex_chp_"+str(k))).Pi #.getAttr("Pi")
            
        else: 
            model.computeIIS()
            print('\nConstraints:')        
            for c in model.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
            print('\nBounds:')
            for v in model.getVars():
                if v.IISLB > 0 :
                    print('Lower bound: %s' % v.VarName)
                elif v.IISUB > 0:
                    print('Upper bound: %s' % v.VarName)
    except GurobiError:
        print('Error in masterproblem')
        
    if final_iteration:
        return (r_obj, r_hp, r_chp)
    else:
        return (r_obj, r_pi, r_sigma_hp, r_sigma_chp)
      
def plot_optimize(day, prop_chp, k_chp, prop_hp):
    """
    
    """
    t = day * 24 * 4
    final_iteration=True
    
    timesteps = Constants.timesteps
    max_time = Constants.max_time_master
    chp_number = len(k_chp[0])
    
    hp_number  = len(prop_hp[0])
    chp_number_props = len(k_chp)
    hp_number_props = len(prop_hp)
    
    P_ren = 	renewables.compute_renewables(rtxt.read_single_indexed_file(os.getcwd()   + "/input_data/wind_speed.txt")[t:t+timesteps],
						      rtxt.read_single_indexed_file(os.getcwd()   + "/input_data/sun_direct.txt")[t:t+timesteps],
						      hp_number, chp_number)
        
    dt = Constants.dt
    k_el = Constants.k_el
    r_el = Constants.r_el
    
    # Gurobi optimization model
    try:
        # Create a new model
        model = Model("masterproblem")
        model.Params.OutputFlag = 0

        # Create variables with one or more sets:
        l_chp = {} # Weighting variables of the CHP proposals
        l_hp  = {} # Weighting variables of the HP proposals
        
        P_imp = {} # Imported electricity
        P_exp = {} # Exported electricity
   
        if final_iteration:
            for p in xrange(hp_number_props):
                for j in xrange(hp_number):
                    l_hp[p,j] = model.addVar(vtype=GRB.BINARY, name="l_hp_"+str(p)+"_"+str(j))
                    
            for p in xrange(chp_number_props):
                for k in xrange(chp_number):
                    l_chp[p,k] = model.addVar(vtype=GRB.BINARY, name="l_chp_"+str(p)+"_"+str(k))
        else:
            for p in xrange(hp_number_props):
                for j in xrange(hp_number):
                    l_hp[p,j] = model.addVar(vtype=GRB.CONTINUOUS, name="l_hp_"+str(p)+"_"+str(j), lb=0.0, ub=1.0)
                    
            for p in xrange(chp_number_props):
                for k in xrange(chp_number):
                    l_chp[p,k] = model.addVar(vtype=GRB.CONTINUOUS, name="l_chp_"+str(p)+"_"+str(k), lb=0.0, ub=1.0)
	
        for t in xrange(timesteps):
            P_imp[t] = model.addVar(vtype=GRB.CONTINUOUS, name="P_imp_"+str(t), lb=0.0)
            P_exp[t] = model.addVar(vtype=GRB.CONTINUOUS, name="P_exp_"+str(t), lb=0.0)

        # Integrate new variables into the model
        model.update()    
	
        # Set objective
        costs_electricity = quicksum(P_imp[t]*k_el - P_exp[t]*r_el for t in range(timesteps)) *dt
        costs_gas_chp = quicksum(quicksum(k_chp[p,k] * l_chp[p,k] for p in range(chp_number_props)) for k in range(chp_number))     
        model.setObjective(costs_electricity + costs_gas_chp, GRB.MINIMIZE)
        
        # Add constraints
        # Electricity balance:
        for t in xrange(timesteps):
            hp_proposal  = quicksum(quicksum(prop_hp[p,j,t]  * l_hp[p,j]  for p in range(hp_number_props))  for j in range(hp_number))
            chp_proposal = quicksum(quicksum(prop_chp[p,k,t] * l_chp[p,k] for p in range(chp_number_props)) for k in range(chp_number))
            model.addConstr(hp_proposal + chp_proposal + P_ren[t] + P_imp[t] - P_exp[t] ==  0 , "ElectricityBalance_"+str(t))
        
        # Convexity constraints
        for j in xrange(hp_number):
            model.addConstr(quicksum(l_hp[p,j] for p in range(hp_number_props)) == 1, "Convex_hp_"+str(j))
        for k in xrange(chp_number):
            model.addConstr(quicksum(l_chp[p,k] for p in range(chp_number_props)) == 1, "Convex_chp_"+str(k))
        
        # Set Gurobi parameters
        model.Params.Presolve = 0
#        model.Params.MIPGap = 0.01
	if not final_iteration:
	    model.Params.TimeLimit = max_time
	else:
	    model.Params.TimeLimit = 15
        
        # Run model
        model.optimize()
        
        # Print final solution
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            r_obj = model.ObjVal
            print("Current objective of the master problem: " + str(r_obj))
            
            if final_iteration:
                r_hp = np.zeros((hp_number))
                r_chp = np.zeros((chp_number))
                
                for p in xrange(hp_number_props):
                    for j in xrange(hp_number):
                        if round(l_hp[p,j].X)==1:
			    r_hp[j] = p
                for p in xrange(chp_number_props):
                    for k in xrange(chp_number):
                        if round(l_chp[p,k].X)==1:
			    r_chp[k] = p
            else:
                r_sigma_hp  = np.zeros(hp_number)
                r_sigma_chp = np.zeros(chp_number)
                r_pi = np.zeros(timesteps)
                            
                for t in xrange(timesteps):
                    r_pi[t] = (model.getConstrByName("ElectricityBalance_"+str(t))).Pi #.getAttr("Pi")
    
                for j in xrange(hp_number):
                    r_sigma_hp[j]  = (model.getConstrByName("Convex_hp_"+str(j))).Pi #.getAttr("Pi")
                    
                for k in xrange(chp_number):
                    r_sigma_chp[k] = (model.getConstrByName("Convex_chp_"+str(k))).Pi #.getAttr("Pi")
            
        else: 
            model.computeIIS()
            print('\nConstraints:')        
            for c in model.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
            print('\nBounds:')
            for v in model.getVars():
                if v.IISLB > 0 :
                    print('Lower bound: %s' % v.VarName)
                elif v.IISUB > 0:
                    print('Upper bound: %s' % v.VarName)
    except GurobiError:
        print('Error in masterproblem')
        
    if final_iteration:
        return (r_obj, r_hp, r_chp)
    else:
        return (r_obj, r_pi, r_sigma_hp, r_sigma_chp)