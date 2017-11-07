# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 22:00:29 2014

@author: T_ohne_admin
"""

from gurobipy import *
import numpy as np
from constants import *
from IPython.core.debugger import Tracer

def optimize(pass_chp, marginals, final):
    """
    
    """
    
    sigma = pass_chp.sigma
    eta   = pass_chp.eta
    Q_nom = pass_chp.Q_nom
    Q_sec_nom = pass_chp.Q_sec_nom
    eta_sec   = pass_chp.eta_sec
    sto_m  = pass_chp.sto_m
    sto_UA = pass_chp.sto_UA
    T_max = pass_chp.t_max
    T_min = pass_chp.t_flow
    t_amb = pass_chp.t_amb
    t_ini = pass_chp.t_init
    q_dem = pass_chp.heatdemand    
    c_w = pass_chp.c_w
    dt  = pass_chp.dt    
    k_g = pass_chp.k_g
    timesteps = pass_chp.timesteps
    P_dem = pass_chp.P_dem
    
    # Gurobi optimization model
    try:
        # Create a new model
        model = Model("chp_subsystem")
	model.Params.OutputFlag = 0
	
        # Create variables with one or more sets:
        x = np.empty([timesteps], dtype=object) # Status CHP (primary heater)
        y = np.empty([timesteps], dtype=object) # Status CHP (secondary heater)
        
        T = np.empty([timesteps], dtype=object) # CHP storage temperature    
        Q = np.empty([timesteps], dtype=object) # Boiler's heat output
        #P = {} # Exported electricity
        
        for t in range(timesteps):
            x[t] = model.addVar(vtype=GRB.BINARY, name="x_chp_"+str(t))
            y[t] = model.addVar(vtype=GRB.BINARY, name="y_chp_"+str(t))
            T[t] = model.addVar(vtype=GRB.CONTINUOUS, name="T_chp_"+str(t), lb=T_min[t], ub=T_max)
            Q[t] = model.addVar(vtype=GRB.CONTINUOUS, name="Q_B_"  +str(t), lb=0.0, ub=Q_sec_nom)
            #P[t] = model.addVar(vtype=GRB.CONTINUOUS, name="P_chp_"+str(t))
       
        # Integrate new variables into the model
        model.update()    
    
        # Set objective
        costs = quicksum((Q_nom*sigma/eta*x[t] + Q[t]/eta_sec)*k_g*dt + marginals[t]*(P_dem[t] - x[t] * Q_nom * sigma) for t in range(timesteps))
        model.setObjective(costs, GRB.MINIMIZE)
        
        # Add constraints
        for t in range(timesteps):
            # Electricity balance:
            #model.addConstr(P[t] == P_dem[t] - x[t] *Q_nom*sigma, "ElectricityBalance_"+str(t))
            
            # Boiler's heat output
            model.addConstr(Q[t] >= y[t] * Q_sec_nom * 0.3, "CHP_Boiler_eq1_"+str(t))
            model.addConstr(Q[t] <= y[t] * Q_sec_nom,       "CHP_Boiler_eq2_"+str(t))
            #model.addConstr(Q[t] == y[t] * Q_sec_nom, "CHP_Boiler_eq_"+str(t))
            
            # Storage equations
            if t == 0:
                t_prev = t_ini
            else:
                t_prev = T[t-1]
            model.addConstr(sto_m * c_w / dt * (T[t] - t_prev) == x[t] * Q_nom - q_dem[t] - sto_UA * (T[t] - t_amb) + Q[t], "Storage_CHP_"+str(t))
        
        '''if GlobVar.inner_count == 1:
	    for t in xrange(timesteps):
		x[t].Start = pass_chp.pricing_res_x[GlobVar.overall_count-2, t]
		y[t].Start = pass_chp.pricing_res_y[GlobVar.overall_count-2, t]
		T[t].Start = pass_chp.pricing_res_T[GlobVar.overall_count-2, t]
		#P[t].Start = pass_chp.res_P[GlobVar.outer_count-1, -1, t]
		Q[t].Start = pass_chp.pricing_res_Q[GlobVar.overall_count-2, t]
	if GlobVar.inner_count > 1:
	    for t in xrange(timesteps):
		x[t].Start = pass_chp.pricing_res_x[GlobVar.overall_count-1, t]
		y[t].Start = pass_chp.pricing_res_y[GlobVar.overall_count-1, t]
		T[t].Start = pass_chp.pricing_res_T[GlobVar.overall_count-1, t]
		#P[t].Start = pass_chp.res_P[GlobVar.outer_count, GlobVar.inner_count-1, t]
		Q[t].Start = pass_chp.pricing_res_Q[GlobVar.overall_count-1, t]'''
	
	if final and GlobVar.final_count > 0:
	    for i in range(GlobVar.final_count):
		x_sol = pass_chp.final_res_x[i].astype(bool)
		y_sol = pass_chp.final_res_y[i].astype(bool)
		model.addConstr(quicksum(x[x_sol]) - quicksum(x[np.invert(x_sol)]) + quicksum(y[y_sol]) - quicksum(y[np.invert(y_sol)]) <= (len(x[x_sol]) + len(y[y_sol]) - 1), "Solution_"+str(i))
	
        # Set Gurobi parameters
        if final:
	    model.Params.MIPGap = Constants.final_MIPGap
	    model.Params.TimeLimit = 5
	    model.Params.Seed = GlobVar.final_count
	else:
	    model.Params.MIPGap = Constants.pricing_MIPGap
	    model.Params.TimeLimit = pass_chp.max_time_subs
        
        # Run model
        model.optimize()
	    
        # Print final solution
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            r_obj = model.ObjVal
    
            r_x = np.zeros(timesteps)
            r_y = np.zeros(timesteps)
            r_T = np.zeros(timesteps)
            r_P = np.zeros(timesteps)
            r_Q = np.zeros(timesteps)
            
            for t in range(timesteps):
                r_x[t] = round(x[t].X)
                r_y[t] = round(y[t].X)
                r_T[t] = T[t].X
                r_P[t] = P_dem[t] - x[t].X * Q_nom * sigma
                r_Q[t] = Q[t].X
                
            # Compute "costs" of the proposal:
            k_prop = k_g * dt * np.sum(r_x*Q_nom *sigma/eta + r_Q/eta_sec)
            
            '''failure = False
            if GlobVar.final_count > 1:
		for i in range(GlobVar.final_count):
		    for j in range(i+1, GlobVar.final_count):
			xcomp = np.array_equal(pass_chp.final_res_x[i], pass_chp.final_res_x[j])
			ycomp = np.array_equal(pass_chp.final_res_y[i], pass_chp.final_res_y[j])
			if xcomp and ycomp:
			    failure = True
			    break
		    if failure:
			break
            
            if final and GlobVar.final_count > 0 and failure:
		for i in range(GlobVar.final_count):
		    x_sol = pass_chp.final_res_x[i].astype(bool)
		    y_sol = pass_chp.final_res_y[i].astype(bool)
		    print x_sol
		    print y_sol
		    print LinExpr(quicksum(x[x_sol]) - quicksum(x[np.invert(x_sol)]) + quicksum(y[y_sol]) - quicksum(y[np.invert(y_sol)])).getValue(),(len(x[x_sol]) + len(y[y_sol]))
		    print quicksum(x[x_sol]).getValue(), quicksum(x[np.invert(x_sol)]).getValue(), quicksum(y[y_sol]).getValue(), quicksum(y[np.invert(y_sol)]).getValue(), len(x[x_sol]), len(y[y_sol]), pass_chp.final_res_obj[i]
		    xl = []
		    yl = []
		    for j in range(timesteps):
			if r_x[j].astype(bool) != pass_chp.final_res_x[i,j].astype(bool):
			    xl.append(j)
			if r_y[j].astype(bool) != pass_chp.final_res_y[i,j].astype(bool):
			    yl.append(j)
		    print xl
		    print yl
		print r_x
		print r_x.astype(int)
		print r_y
		print r_y.astype(int)
		Tracer()()'''
        
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
        print('Error in CHP subproblem '+ str(pass_chp.nr))
        
    #print r_P
    #print P_dem
    #print r_x
    #print r_y
    #print r_T
    #print T_min
    #print r_Q
    #print k_prop
    #raw_input()
    
    return (r_obj, r_x, r_y, r_T, - r_P, r_Q, k_prop)