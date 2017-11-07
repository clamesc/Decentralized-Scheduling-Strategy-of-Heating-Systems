# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 20:45:37 2014

@author: T_ohne_admin
"""

from gurobipy import *
import numpy as np
from constants import *
from IPython.core.debugger import Tracer
import random

def optimize(pass_hp, marginals, final):
    """
    
    """
   
    Q_nom = pass_hp.Q_prim
    P_nom = pass_hp.P_prim
    Q_sec_nom = pass_hp.Q_sec_nom
    eta_sec   = pass_hp.eta_sec
    sto_m  = pass_hp.sto_m
    sto_UA = pass_hp.sto_UA
    T_max = pass_hp.t_max
    T_min = pass_hp.t_flow
    t_amb = pass_hp.t_amb
    t_ini = pass_hp.t_init
    q_dem = pass_hp.heatdemand    
    c_w = pass_hp.c_w
    dt  = pass_hp.dt
    P_dem = pass_hp.P_dem
    
    timesteps = pass_hp.timesteps
   
    # Gurobi optimization model
    try:
        # Create a new model
        model = Model("hp_subsystem")
	model.Params.OutputFlag = 0
	
        # Create variables
        x = np.empty([timesteps], dtype=object) # Status HP (primary heater)
        y = np.empty([timesteps], dtype=object) # Status HP  (secondary heater)
        T = np.empty([timesteps], dtype=object) # HP storage temperature    
        #P = np.empty([timesteps], dtype=object) # Exported electricity
        
        for t in range(timesteps):
            x[t] = model.addVar(vtype=GRB.BINARY, name="x_hp_"    +str(t))
            y[t] = model.addVar(vtype=GRB.BINARY, name="y_hp_"    +str(t))
            T[t] = model.addVar(vtype=GRB.CONTINUOUS, name="T_hp_"+str(t), lb=T_min[t], ub=T_max)
            #P[t] = model.addVar(vtype=GRB.CONTINUOUS, name="P_hp_" +str(t), lb=0.0)
       
        # Integrate new variables into the model
        model.update()
    
        # Set objective
        model.setObjective(quicksum((x[t] * P_nom[t] + Q_sec_nom / eta_sec * y[t] + P_dem[t])*marginals[t] for t in range(timesteps)), GRB.MINIMIZE)
        
        # Add constraints
        for t in xrange(timesteps):
            # Electricity balance:
            #model.addConstr(P[t] == x[t] * P_nom[t] + Q_sec_nom / eta_sec * y[t] + P_dem[t], "ElectricityBalance_"+str(t))
            
            # Storage equations
            if t == 0:
                t_prev = t_ini
            else:
                t_prev = T[t-1]
            model.addConstr(sto_m * c_w / dt * (T[t] - t_prev) == x[t] * Q_nom[t] - q_dem[t] - sto_UA * (T[t] - t_amb) + y[t] * Q_sec_nom, "Storage_HP_"+str(t))
        
        '''if GlobVar.inner_count == 1:
	    for t in xrange(timesteps):
		x[t].Start = pass_hp.pricing_res_x[GlobVar.overall_count-2, t]
		y[t].Start = pass_hp.pricing_res_y[GlobVar.overall_count-2, t]
		T[t].Start = pass_hp.pricing_res_T[GlobVar.overall_count-2, t]
		#P[t].Start = pass_hp.res_P[GlobVar.outer_count-1, -1, t]
	elif GlobVar.inner_count > 1:
	    for t in xrange(timesteps):
		x[t].Start = pass_hp.pricing_res_x[GlobVar.overall_count-1, t]
		y[t].Start = pass_hp.pricing_res_y[GlobVar.overall_count-1, t]
		T[t].Start = pass_hp.pricing_res_T[GlobVar.overall_count-1, t]
		#P[t].Start = pass_hp.res_P[GlobVar.outer_count-1, -1, t]'''
	
	if final and GlobVar.final_count > 0:
	    for i in range(GlobVar.final_count):
		x_sol = pass_hp.final_res_x[i].astype(bool)
		y_sol = pass_hp.final_res_y[i].astype(bool)
		model.addConstr(quicksum(x[x_sol]) - quicksum(x[np.invert(x_sol)]) + quicksum(y[y_sol]) - quicksum(y[np.invert(y_sol)]) <= len(x[x_sol]) + len(y[y_sol]) -1, "Solution_"+str(i))
	
        if final:
	    model.Params.MIPGap = Constants.final_MIPGap
	    model.Params.TimeLimit = 5
	    model.Params.Seed = GlobVar.final_count+1
	else:
	    model.Params.MIPGap = Constants.pricing_MIPGap
	    model.Params.TimeLimit = pass_hp.max_time_subs
        
        
        # Run model
        model.optimize()
    
        # Print final solution
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            r_obj = model.ObjVal
    
            r_x = np.zeros(timesteps)
            r_y = np.zeros(timesteps)
            r_T = np.zeros(timesteps)
            r_P = np.zeros(timesteps)
            
            for t in range(timesteps):
                r_x[t] = round(x[t].X)
                r_y[t] = round(y[t].X)
                r_T[t] = T[t].X
                r_P[t] = r_x[t] * P_nom[t] + Q_sec_nom / eta_sec * r_y[t] + P_dem[t]

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
        print('Error in HP subproblem ' + str(pass_hp.nr))
        
    #print r_x
    #print r_y
    #print r_T
    #print r_P
    #raw_input()
        
    return r_obj, r_x, r_y, r_T, - r_P