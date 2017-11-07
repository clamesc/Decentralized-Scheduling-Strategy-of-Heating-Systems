# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 21:39:21 2014

@author: T_ohne_admin 
# Solve masterproblem with binary restrictions
(obj, lambda_chp, lambda_hp) = mp.finalize()

# Retrieve optimal schedules from each house:
x_hp = np.zeros((Constants.timesteps, len(data.hp_nom)))
P_hp = np.zeros((Constants.timesteps, len(data.hp_nom)))
for i in xrange(len(data.hp_nom)):
    (temp_x, temp_y, temp_T, temp_P) = hp[i].get_optimal_schedule(lambda_hp[:,i])
    x_hp[:,i] = temp_x
    P_hp[:,i] = temp_P

x_chp = np.zeros((Constants.timesteps, len(data.chp_nom)))    
P_chp = np.zeros((Constants.timesteps, len(data.chp_nom)))
for j in xrange(len(data.chp_nom)):
    (temp_x, temp_y, temp_T, temp_P, temp_Q) = chp[j].get_optimal_schedule(lambda_chp[:,j])
    x_chp[:,j] = temp_x
    P_chp[:,j] = temp_P"""

import matplotlib.pyplot as plt
import h5py
import time
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib
import matplotlib.colors as mcol
from constants import *
from IPython.core.debugger import Tracer
import model_masterproblem
import basic_functions.read_txt as rtxt
import basic_functions.heating_tech
import heat_pumps.HP_characteristics
import microgrid_functions.renewables as renewables
from matplotlib.ticker import MaxNLocator

def read(filename, name, indexed=True):
    d = filename[name]
    data = np.zeros(d.shape)
    d.read_direct(data)
    if indexed:
	return data[data != 0]
    else:
	return data
      
def mread(filename, name):
    d = filename[name]
    data = np.zeros(d.shape)
    d.read_direct(data)
    return data
  
def remove_zeros(arr):
    i = 0
    while i < len(arr):
	if arr[i].all() == 0:
	    arr = np.delete(arr, i, 0)
	    i -= 1
	i += 1
    return arr

def plot1a():
    values = []
    for day in range(0,364):

	lr_file = h5py.File(Constants.path + "/lr_"+str(day)+".hdf5")
	cg_file = h5py.File(Constants.path + "/cg_"+str(day)+".hdf5")
	int_file = h5py.File(Constants.path + "/int_obj_value_"+str(day)+".hdf5")
	
	values.append([read(lr_file, "lin_obj_values")[-1], 
		       np.concatenate((read(lr_file, "res_lr_bounds_master"),read(lr_file, "res_lr_bounds_sugradient")), axis=0)[-1], 
		       read(lr_file, "int_obj_values")[-1], 
		       read(cg_file, "lin_obj_values")[-1],
		       read(int_file, "cg_int_obj_value", False),
		       read(int_file, "lr_int_obj_value", False),
		       read(cg_file, "res_lr_bounds_master")[-1]])
	
	lr_file.close()
	cg_file.close()
	int_file.close()

    values = np.array(values)
    
    plt.rc('figure', figsize=(11.69,6))
    #f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    
    #l1, = plt.plot(values[:,0], ((values[:,1]-values[:,0])/values[:,0]), 'r.')
    ##l1, = plt.plot(values[:,0], (values[:,1]-values[:,0]), 'r.')
    #l2, = plt.plot(values[:,0], ((values[:,2]-values[:,0])/values[:,0]), 'k.')
    ##l2, = plt.plot(values[:,0], (values[:,2]-values[:,0]), 'k.')
    #l3, = ax1.plot(values[:,0], ((values[:,3]-values[:,0])/values[:,0]), 'bo', markersize=8)
    ##l3, = plt.plot(values[:,0], (values[:,3]-values[:,0]), 'b.')
    l4, = plt.plot(values[:,0], ((values[:,4]-values[:,0])/values[:,0]), 'b^', markersize = 8)
    l5, = plt.plot(values[:,0], ((values[:,5]-values[:,0])/values[:,0]), 'rv', markersize = 8)
    #l6, = plt.plot(values[:,0], ((values[:,6]-values[:,0])/values[:,0]), 'b.')
    
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize=16)
    plt.yticks([0.00, 0.01, 0.02, 0.03])
    
    plt.ylim([-0.01,0.04])
    plt.xlabel(r'Linear Solution (Combined Algorithm: $z_{LRDW}$) [Euro]', fontsize = 20)
    #ax1.set_ylabel(r'Relative deviation', fontsize = 20)
    plt.ylabel(r'Relative deviation', fontsize = 20)
    #ax1.legend([l3], [r'CG: $z_{LRDW}$'], frameon = False, fontsize = 20, numpoints=1)
    #ax2.legend([l4, l5], [r'CG: $z_{RDW}$', r'LR: $z_{RDW}$ (Pricing proposals)'], frameon = False, fontsize = 20, numpoints=1)
    plt.legend([l4, l5], [r'CG: $z_{RDW}$', r'CGLR: $z_{RDW}$'], frameon = False, fontsize = 20, numpoints=1)
    plt.subplots_adjust(hspace=0, left=0.08, right=0.97)
    #ax1.grid(True)
    plt.grid(True)
    plt.show()
    
    # 1 und 6 - LBs
    # 4 und 5 - CG int
    # 2 und 5 - LR int
    
def plot1b():
    values = []
    for day in range(0,364):

	lr_file = h5py.File(Constants.path + "/lr_"+str(day)+".hdf5")
	cg_file = h5py.File(Constants.path + "/cg_"+str(day)+".hdf5")
	int_file = h5py.File(Constants.path + "/int_obj_value_"+str(day)+".hdf5")
	
	values.append([read(lr_file, "lin_obj_values")[-1], 
		       np.concatenate((read(lr_file, "res_lr_bounds_master"),read(lr_file, "res_lr_bounds_sugradient")), axis=0)[-1], 
		       read(lr_file, "int_obj_values")[-1], 
		       read(cg_file, "lin_obj_values")[-1],
		       read(int_file, "cg_int_obj_value", False),
		       read(int_file, "lr_int_obj_value", False),
		       read(cg_file, "res_lr_bounds_master")[-1]])
	
	lr_file.close()
	cg_file.close()
	int_file.close()

    plt.rc('figure', figsize=(11.69,3.5))

    
    values = np.array(values)
    l1, = plt.plot(values[:,0], ((values[:,1]-values[:,0])/values[:,0]), 'r<')
    ##l1, = plt.plot(values[:,0], (values[:,1]-values[:,0]), 'r.')
    #l2, = plt.plot(values[:,0], ((values[:,2]-values[:,0])/values[:,0]), 'k.')
    ##l2, = plt.plot(values[:,0], (values[:,2]-values[:,0]), 'k.')
    #l3, = plt.plot(values[:,0], ((values[:,3]-values[:,0])/values[:,0]), 'b.')
    ##l3, = plt.plot(values[:,0], (values[:,3]-values[:,0]), 'b.')
    #l4, = plt.plot(values[:,0], ((values[:,4]-values[:,0])/values[:,0]), 'g.')
    #l5, = plt.plot(values[:,0], ((values[:,5]-values[:,0])/values[:,0]), 'b.')
    l6, = plt.plot(values[:,0], ((values[:,6]-values[:,0])/values[:,0]), 'b>')
    
    plt.xlabel(r'LR: $z_{LRDW}$ [Euro]', fontsize = 14)
    plt.ylabel(r'Relative deviation', fontsize = 14)
    plt.legend([l1, l6], [r'LR: $z_{LR}$', r'CG: $z_{LR}$'], frameon = False, loc=4, fontsize = 14, numpoints=1)
    plt.grid(True)
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.subplots_adjust(left=0.08, right=0.97, bottom=0.17)
    plt.show()
    
    # 1 und 6 - LBs
    # 4 und 5 - CG int
    # 2 und 5 - LR int

def plot1c():
    values = []
    for day in range(0,364):

	lr_file = h5py.File(Constants.path + "/lr_"+str(day)+".hdf5")
	cg_file = h5py.File(Constants.path + "/cg_"+str(day)+".hdf5")
	int_file = h5py.File(Constants.path + "/int_obj_value_"+str(day)+".hdf5")
	
	values.append([read(lr_file, "lin_obj_values")[-1], 
		       np.concatenate((read(lr_file, "res_lr_bounds_master"),read(lr_file, "res_lr_bounds_sugradient")), axis=0)[-1], 
		       read(lr_file, "int_obj_values")[-1], 
		       read(cg_file, "lin_obj_values")[-1],
		       read(int_file, "cg_int_obj_value", False),
		       read(int_file, "lr_int_obj_value", False),
		       read(cg_file, "res_lr_bounds_master")[-1]])
	
	lr_file.close()
	cg_file.close()
	int_file.close()

    plt.rc('figure', figsize=(11.69,3.5))
    
    values = np.array(values)
    #l1, = plt.plot(values[:,0], ((values[:,1]-values[:,0])/values[:,0]), 'r<')
    ##l1, = plt.plot(values[:,0], (values[:,1]-values[:,0]), 'r.')
    l2, = plt.plot(values[:,0], ((values[:,2]-values[:,0])/values[:,0]), 'gD', markersize=4)
    ##l2, = plt.plot(values[:,0], (values[:,2]-values[:,0]), 'k.')
    #l3, = plt.plot(values[:,0], ((values[:,3]-values[:,0])/values[:,0]), 'b.')
    ##l3, = plt.plot(values[:,0], (values[:,3]-values[:,0]), 'b.')
    l4, = plt.plot(values[:,0], ((values[:,4]-values[:,0])/values[:,0]), 'b^', markersize=5)
    l5, = plt.plot(values[:,0], ((values[:,5]-values[:,0])/values[:,0]), 'rv', markersize=5)
    #l6, = plt.plot(values[:,0], ((values[:,6]-values[:,0])/values[:,0]), 'b>')
    
    plt.xlabel(r'LR: $z_{LRDW}$ [Euro]', fontsize = 14)
    plt.ylabel(r'Relative deviation', fontsize = 14)
    plt.legend([l2, l5, l4], [r'LR: $z_{RDW}$ (Final proposals)', r'LR: $z_{RDW}$ (Pricing proposals)', r'CG: $z_{RDW}$'], frameon = False, fontsize = 14, numpoints=1)
    plt.grid(True)
    plt.subplots_adjust(left=0.08, right=0.97, bottom=0.17)
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.show()
    
    # 1 und 6 - LBs
    # 4 und 5 - CG int
    # 2 und 5 - LR int

def compute2():
    for day in range(0,366):
	cg_file = h5py.File(Constants.path + "/cg_"+str(day)+".hdf5")
	
	cg_costs_chp = mread(cg_file, "pricing_costs_chp")
	cg_prop_chp = mread(cg_file, "pricing_proposals_chp")
	cg_prop_hp = mread(cg_file, "pricing_proposals_hp")
	
	cg_int_obj_value= model_masterproblem.plot_optimize(day, cg_prop_chp, cg_costs_chp, cg_prop_hp)[0]

	lr_file = h5py.File(Constants.path + "/lr_"+str(day)+".hdf5")

	lr_p_pro_chp = mread(lr_file, "pricing_proposals_chp")
	lr_p_costs_chp = mread(lr_file, "pricing_costs_chp")
	lr_p_pro_hp = mread(lr_file, "pricing_proposals_hp")
	
	lr_int_obj_value = model_masterproblem.plot_optimize(day, lr_p_pro_chp, lr_p_costs_chp, lr_p_pro_hp)[0]
	
	f = h5py.File(Constants.path + "/int_obj_value_" + str(day) + ".hdf5", "w")
	
	f.create_dataset("cg_int_obj_value", data = cg_int_obj_value) 
	f.create_dataset("lr_int_obj_value", data = lr_int_obj_value) 
	
	f.close()
	
def compute3():
    for day in range(0,366):
	cg_file = h5py.File(Constants.path + "/cg_"+str(day)+".hdf5")
	
	cg_costs_chp = mread(cg_file, "pricing_costs_chp")[-1]
	cg_prop_chp = mread(cg_file, "pricing_proposals_chp")[-1]
	cg_prop_hp = mread(cg_file, "pricing_proposals_hp")[-1]
	
	cg_int_obj_value= model_masterproblem.plot_optimize(day, cg_prop_chp, cg_costs_chp, cg_prop_hp)[0]
	
	f = h5py.File(Constants.path + "/int_single_value_" + str(day) + ".hdf5", "w")
	
	f.create_dataset("cg_int_obj_value", data = cg_int_obj_value) 
	
	f.close()

def plot2():
    values=[]
    for day in range(0,364):
	lr_file = h5py.File(Constants.path + "/lr_"+str(day)+".hdf5")
	values.append(read(lr_file, "lin_obj_values")[-1])
    plt.rc('figure', figsize=(11.69,3.5))
    l1, = plt.plot(range(1,365), values, 'k-', linewidth = 2.0)
    plt.xlabel(r'Time [days]', fontsize = 14)
    plt.ylabel(r'LR: $z_{LRDW}$ [Euro]', fontsize = 14)
    plt.grid(True)
    plt.xlim([1,365])
    plt.subplots_adjust(left=0.08, right=0.97, bottom=0.17)
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.show()
    
def plot3():
    values=[]
    for day in range(0,364):
	cg_file = h5py.File(Constants.path + "/cg_"+str(day)+".hdf5")
	lr_file = h5py.File(Constants.path + "/lr_"+str(day)+".hdf5")
	values.append([read(cg_file, "sub_time")[-1],
		       read(lr_file, "lin_obj_values")[-1],
		       max(max(read(lr_file, "sub_time")),max(read(lr_file,"master_time")))])
    values = np.array(values)
    plt.rc('figure', figsize=(11.69,3.5))
    l1, = plt.plot(values[:,1], values[:,0], 'b^', markersize=8)
    l2, = plt.plot(values[:,1], values[:,2], 'rv', markersize=8)
    plt.legend([l1, l2], [r'CG', r'CGLR'], frameon = True, fontsize = 20, numpoints=1)
    plt.ylabel(r'Time [seconds]', fontsize = 20)
    plt.xlabel(r'CGLR: $z_{LRDW}$ [Euro]', fontsize = 20)
    plt.subplots_adjust(left=0.08, right=0.97, bottom=0.17)
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.show()

def plot4(day):
    lr_file = h5py.File(Constants.path + "/lr_"+str(day)+".hdf5")
    x_hp = read(lr_file, "final_hp", False)
    x_chp = read(lr_file, "final_chp", False)
    chp_input_data = 	np.transpose(rtxt.read_multiple_indexed_file(os.getcwd() + "/input_data/input_data_chp.csv", delimiter = ";"))
    hp_input_data = 	np.transpose(rtxt.read_multiple_indexed_file(os.getcwd() + "/input_data/input_data_hp.csv", delimiter = ";"))
    t_ambient = 	rtxt.read_single_indexed_file(os.getcwd()  + "/input_data/temperature.txt")[(day*24*4):((day*24*4)+192)]
    t_flow = basic_functions.heating_tech.heatingCurve(t_ambient+273.15) - 273.15
    p_res = renewables.compute_renewables(rtxt.read_single_indexed_file(os.getcwd()   + "/input_data/wind_speed.txt")[(day*24*4):((day*24*4)+192)],
							      rtxt.read_single_indexed_file(os.getcwd()   + "/input_data/sun_direct.txt")[(day*24*4):((day*24*4)+192)],
							      51, 51)
    
    lr_p_pro_chp = mread(lr_file, "pricing_proposals_chp")
    lr_p_costs_chp = mread(lr_file, "pricing_costs_chp")
    lr_p_pro_hp = mread(lr_file, "pricing_proposals_hp")
    lr_int_obj_value, r_hp, r_chp = model_masterproblem.plot_optimize(day, lr_p_pro_chp, lr_p_costs_chp, lr_p_pro_hp)

    qnom = []
    sigma = []
    p_hp = []
    for i in range(len(chp_input_data)):
	    p_hp.append(heat_pumps.HP_characteristics.get_hp_data(hp_input_data[i,0], t_ambient, t_flow)[1])
	    qnom.append(chp_input_data[i,0])
	    sigma.append(chp_input_data[i,1])
    values_hp = np.zeros(192)
    values_chp = np.zeros(192)
    values_ie = np.zeros(192)
    values_ie2 = np.zeros(192)
    for i in range(51):
	values_hp += [read(lr_file,"HP_"+str(i)+"/x_f", False)[x_hp[i]][j]*p_hp[i][j] for j in range(192)]
	values_chp += read(lr_file, "CHP_"+str(i)+"/x_f", False)[x_chp[i]]*sigma[i]*qnom[i]
	values_ie += read(lr_file, "CHP_"+str(i)+"/P_f", False)[x_chp[i]]
	values_ie += read(lr_file, "HP_"+str(i)+"/P_f", False)[x_hp[i]]
	values_ie2 += read(lr_file, "HP_"+str(i)+"/P_p", False)[int(r_hp[i])]
	values_ie2 += read(lr_file, "CHP_"+str(i)+"/P_p", False)[int(r_chp[i])]
    values_ie += p_res
    values_ie2 += p_res
    values_ie = [values_ie[i]*0.001 for i in range(len(values_ie))]
    values_ie2 = [values_ie2[i]*0.001 for i in range(len(values_ie2))]
    
    
    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    plt.rc('figure', figsize=(11.69,6))
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
    l1, = ax2.step([i*0.25 for i in range(192)], values_ie2, 'k-', linewidth=2.0)
    l3, = ax1.step([i*0.25 for i in range(192)], values_ie, 'k-', linewidth=2.0)

    plt.xticks([0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48], fontsize = 12)
    plt.xlim([0,47.75])
    f.subplots_adjust(hspace=0, left=0.11, right=0.97)
    ax2.set_xlabel("Time [hours]", fontsize=14)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_ylabel("$\mathregular{P_{imp/exp}}$ [kW]", fontsize=14)
    ax2.set_ylabel("$\mathregular{P_{imp/exp}}$ [kW]", fontsize=14)
    plt.figtext(0.005,0.7,"(a)", size = 'large')
    plt.figtext(0.005,0.3,"(b)", size = 'large')
    nbins = len(ax1.get_xticklabels())
    #ax1.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='lower'))
    #ax2.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
    #plt.ylim([-175,175])
    #plt.yticks([-150,-100,-50,0,50,100,150])
    plt.show()

def plot5(day):
    cg_file = h5py.File(Constants.path + "/cg_"+str(day)+".hdf5")
    lr_file = h5py.File(Constants.path + "/lr_"+str(day)+".hdf5")
    int_file = h5py.File(Constants.path + "/int_obj_value_"+str(day)+".hdf5")
    
    cg_master_time = read(cg_file, "master_time")
    cg_sub_time = read(cg_file, "sub_time")
    cg_res_lr_bounds = read(cg_file, "res_lr_bounds_master")
    cg_prop_chp = mread(cg_file, "pricing_proposals_chp")
    cg_costs_chp = mread(cg_file, "pricing_costs_chp")
    cg_prop_hp = mread(cg_file, "pricing_proposals_hp")
    cg_lin_obj_values = read(cg_file, "lin_obj_values")
    
    lr_sub_time = read(lr_file, "sub_time")
    lr_master_time = read(lr_file, "master_time")
    lr_final_time = read(lr_file, "final_time")
    lr_final_pro_chp = mread(lr_file, "final_proposals_chp")
    lr_final_costs_chp = mread(lr_file, "final_costs_chp")
    lr_final_pro_hp = mread(lr_file, "final_proposals_hp")
    lr_lin_obj_values = read(lr_file, "lin_obj_values")
    lr_bounds = read(lr_file, "res_lr_bounds_sugradient")

    lr_int_obj_values = np.array([model_masterproblem.plot_optimize(day, lr_final_pro_chp[:j], lr_final_costs_chp[:j], lr_final_pro_hp[:j])[0] for j in range(1,6)])

    cg_int_obj_value= read(int_file, "cg_int_obj_value", False)
    lr_int_obj_value = read(int_file, "lr_int_obj_value", False)

    #cg_int_single_values = np.array([model_masterproblem.plot_optimize(day, np.array([cg_prop_chp[i,:,:]]), np.array([cg_costs_chp[i,:]]), np.array([cg_prop_hp[i,:,:]]))[0] for i in range(len(cg_lin_obj_values-1))])

    plt.rc('figure', figsize=(11.69,3.5))
    l1, = plt.plot(lr_master_time, 		lr_lin_obj_values, 		'k-', 	linewidth = 3.0)
    l2, = plt.plot(lr_sub_time[:-1], 		lr_bounds, 			'r-', 	linewidth = 3.0)
    #l3, = plt.plot(lr_final_time, 		lr_int_obj_values, 		'gD', markersize=4)
    l4, = plt.plot(cg_master_time, 		cg_lin_obj_values, 		'k--', linewidth = 2.0)
    l5, = plt.plot(cg_sub_time, 		cg_res_lr_bounds, 		'b--', linewidth = 2.0)
    l6, = plt.plot(cg_master_time[-1], 		cg_int_obj_value, 		'b^', markersize=12)
    #l7, = plt.plot(cg_master_time[1:], 		cg_int_single_values[:-1],	'k:')
    l8, = plt.plot(lr_master_time[-1], 			lr_int_obj_value, 		'rv', markersize=12)

    plt.xlim([0,max(cg_master_time[-1]+15,max(lr_final_time)+15)])
    plt.xlabel("Time [seconds]", fontsize=20)
    plt.ylabel("Costs [Euro]", fontsize=20)
    plt.ylim([min(cg_res_lr_bounds)-100,lr_lin_obj_values[0]+100])
    #plt.legend([l1,l2, l3, l8, l4,l5, l6], ["LR: $z_{LRDW}$","LR: $z_{LR}$", "LR: $z_{RDW}$ (Final proposals)", "LR: $z_{RDW}$ (Pricing proposals)", "CG: $z_{LRDW}$","CG: $z_{LR}$","CG: $z_{RDW}$"], loc='center left', fontsize=14, bbox_to_anchor=(1, 0.5), frameon=False, numpoints=1)
    plt.legend([l1,l2, l8, l4,l5, l6], ["CGLR: $z_{LRDW}$","CGLR: $z_{LR}$", "CGLR: $z_{RDW}$", "CG: $z_{LRDW}$","CG: $z_{LR}$","CG: $z_{RDW}$"], loc='center left', fontsize=20, bbox_to_anchor=(1, 0.5), frameon=False, numpoints=1)
    plt.subplots_adjust(left=0.08, right=0.705, bottom=0.17)
    plt.yticks(fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.show()


def plot6(day):
    cg_file = h5py.File(Constants.path + "/cg_"+str(day)+".hdf5")
    lr_file = h5py.File(Constants.path + "/lr_"+str(day)+".hdf5")
    
    lr_marginals = np.concatenate((remove_zeros(mread(lr_file, "marginals_mu_master"))[:1], remove_zeros(mread(lr_file, "marginals_mu_subgradient"))))

    cg_marginals = remove_zeros(mread(cg_file, "marginals_mu_master"))

    timesteps = lr_marginals.shape[1]
    step_length = 0.25

    plt.rc('figure', figsize=(11.69,6))
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

    # First subplot
    cnorm = mcol.Normalize(vmin=0,vmax=len(cg_marginals)-1)
    cpick = cm.ScalarMappable(norm=cnorm,cmap=cm.Blues)
    cpick.set_array([])

    for i in range(len(cg_marginals)):
	ax1.step([x*step_length for x in range(timesteps)], cg_marginals[i]*100*1000, color=cpick.to_rgba(i))

    plt.xlim([0,47.75])
    plt.ylim([1,(29.21 / (100 * 3600 * 1000)*900*100*1000 + (5.00  / (100 * 3600 * 1000)*900*100*1000-1))])
    plt.xticks([0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48])
    plt.figtext(0.005,0.7,"(a)", size = 'large')
    plt.figtext(0.005,0.3,"(b)", size = 'large')
    ax1.set_title("Shadow prices", fontsize=20)
    ax1.set_ylabel("$\mathregular{\pi}$ [ct/kW]", fontsize=20)

    # Second Subplot
    cnorm = mcol.Normalize(vmin=0,vmax=len(lr_marginals)-1)
    cpick = cm.ScalarMappable(norm=cnorm,cmap=cm.Reds)
    cpick.set_array([])
    for i in range(len(lr_marginals)):
	ax2.step([x*step_length for x in range(timesteps)], lr_marginals[i]*100*1000, color=cpick.to_rgba(i))
    
    plt.xlim([0,(timesteps-1)*step_length])
    plt.ylim([1,(29.21 / (100 * 3600 * 1000)*900*100*1000 + (5.00  / (100 * 3600 * 1000)*900*100*1000-1))])
    ax2.set_xlabel("Time [hours]", fontsize=20)
    ax2.set_ylabel("$\mathregular{\pi}$ [ct/kW]", fontsize=20)
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0, left=0.08, right=0.97)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 16)

    plt.show()
    
def show_sol():
    values = []
    for day in range(0,364):

	lr_file = h5py.File(Constants.path + "/lr_"+str(day)+".hdf5")
	cg_file = h5py.File(Constants.path + "/cg_"+str(day)+".hdf5")
	int_file = h5py.File(Constants.path + "/int_obj_value_"+str(day)+".hdf5")
	
	values.append([read(lr_file, "lin_obj_values")[-1], 
		       np.concatenate((read(lr_file, "res_lr_bounds_master"),read(lr_file, "res_lr_bounds_sugradient")), axis=0)[-1], 
		       read(lr_file, "int_obj_values")[-1], 
		       read(cg_file, "lin_obj_values")[-1],
		       read(int_file, "cg_int_obj_value", False),
		       read(int_file, "lr_int_obj_value", False),
		       read(cg_file, "res_lr_bounds_master")[-1]])
	
	lr_file.close()
	cg_file.close()
	int_file.close()

    values = np.array(values)
    values = np.append(values, np.transpose(np.array([range(364)])), 1)
    values = values[np.argsort(values[:,0])]
    np.set_printoptions(threshold=np.nan)
    for i in range(0,365):
	  print str(values[i,0])+"\t"+str(values[i,2])+"\t"+str(int(values[i,7]))