# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 12:34:23 2014

@author: tsz

This file contains a few characteristic curves of real HP units
"""

from __future__ import division
import numpy as np
import basic_functions.heating_tech
from constants import *

class LA12TU(): # http://www.dimplex.de/pdf/de/produktattribute/produkt_1725609_extern_egd.pdf
    t_ambient = np.array([-20, -15, -7, 2, 7, 10, 12, 20])
    t_flow = np.array([35, 45, 55])
    Q = np.array([[4.89, 4.7,   4.5], 
                  [5.87, 5.7,   5.5], 
                  [7.6,  7.35,  7.17],
                  [9.6,  9.1,   8.8],
                  [11.4, 10.85, 9.8],
                  [11.7, 11.2,  10.6],
                  [12.2, 11.4,  10.9],
                  [13.6, 12.8,  12.39]])
                  
    cop = np.array([[1.91, 1.48, 1.20], 
                    [2.28, 1.77, 1.45], 
                    [3.00, 2.30, 1.88],
                    [3.70, 2.84, 2.32],
                    [4.30, 3.42, 2.50],
                    [4.60, 3.53, 2.75],
                    [4.78, 3.56, 2.87],
                    [5.33, 4.06, 3.30]])
               
    P = Q / cop

class LA9TU(): # http://www.dimplex.de/pdf/de/produktattribute/produkt_1725608_extern_egd.pdf
    t_ambient = np.array([-20, -15, -7, 2, 7, 10, 12, 20])
    t_flow = np.array([35, 45, 55])
    Q = np.array([[2.96,   2.26,  2.03], 
                  [3.73,   3.15,  2.78], 
                  [5.50,   4.56,  3.98],
                  [7.50,   6.53,  5.88],
                  [9.20,   8.21,  7.10],
                  [10.20,  8.88,  7.81],
                  [11.45, 10.29,  9.11],
                  [12.70, 11.70, 10.40]])
               
    cop = np.array([[1.86, 1.42, 1.28], 
                    [2.16, 1.79, 1.51], 
                    [2.80, 2.28, 1.87],
                    [3.70, 2.86, 2.45],
                    [4.20, 3.62, 2.70],
                    [4.50, 3.55, 2.95],
                    [4.77, 4.04, 3.31],
                    [5.29, 4.33, 3.48]])
               
    P = Q / cop
  
class LA6TU():
    t_ambient = np.array([-20, -15, -7, 2, 7, 10, 12, 20])
    t_flow = np.array([35, 45, 55])
    Q = np.array([[2.38,   2.28,  1.84], # Q(-20°C, W55) interpoliert
                  [2.97,   2.83,  2.52], # Q(-15°C, W55) interpoliert
                  [4.00,   3.76,  3.61],
                  [5.10,   4.84,  4.66],
                  [6.40,   6.10,  5.68],
                  [6.70,   6.30,  6.00],
                  [7.00,   6.40,  6.20],
                  [8.04,   7.67,  7.29]])
               
    cop = np.array([[1.79, 1.44, 1.04], # COP(-20°C, W55) interpoliert
                    [2.20, 1.75, 1.36], # COP(-15°C, W55) interpoliert
                    [2.90, 2.22, 1.81],
                    [3.80, 2.84, 2.25],
                    [4.60, 3.50, 2.73],
                    [4.70, 3.62, 2.83],
                    [4.93, 3.66, 2.91],
                    [5.66, 4.41, 3.41]])
               
    P = Q / cop
  
class LA17TU():
    t_ambient = np.array([-20, -15, -7, 2, 7, 10, 12, 20])
    t_flow = np.array([35, 45, 55])
    Q = np.array([[4.00,    3.65,   3.30],
                  [4.90,    4.50,   4.00],
                  [5.40,    5.70,   5.35],
                  [8.20,    7.90,   7.60],
                  [10.00,   9.57,   9.20],
                  [10.50,   10.15,  9.80],
                  [11.00,   10.50,  10.10],
                  [13.00,   12.50,  12.00]])
               
    cop = np.array([[1.92, 1.45, 1.09],
                    [2.33, 1.80, 1.31],
                    [3.00, 2.25, 1.76],
                    [3.80, 3.11, 2.49],
                    [4.50, 3.75, 2.80],
                    [4.90, 3.95, 3.18],
                    [5.24, 4.04, 3.26],
                    [6.05, 4.72, 3.81]])
               
    P = Q / cop
  
class LA25TU():
    t_ambient = np.array([-20, -15, -7, 2, 7, 10, 12, 20])
    t_flow = np.array([35, 45, 55])
    Q = np.array([[6.20,   5.50,  4.91],
                  [7.50,   7.00,  6.10],
                  [9.10,   8.46,  7.84],
                  [11.30,  10.70, 10.10],
                  [13.90,  13.16, 12.40],
                  [15.00,  14.06, 13.27],
                  [15.80,  15.00, 14.00],
                  [19.20,  18.00, 17.00]])
               
    cop = np.array([[2.07, 1.49, 1.14],
                    [2.50, 1.92, 1.42],
                    [3.00, 2.29, 1.78],
                    [3.80, 2.85, 2.24],
                    [4.50, 3.46, 2.80],
                    [4.90, 3.61, 2.98],
                    [4.86, 3.85, 3.11],
                    [5.82, 4.50, 3.70]])
               
    P = Q / cop
  
class LA40TU():
    t_ambient = np.array([-20, -15, -7, 2, 7, 10, 12, 20])
    t_flow = np.array([35, 45, 55])
    Q = np.array([[10.78,  10.09, 9.41],
                  [11.83,  11.11, 10.39],
                  [13.50,  12.73, 11.96],
                  [16.80,  16.00, 15.19],
                  [20.00,  18.80, 17.60],
                  [21.70,  19.75, 18.43],
                  [22.67,  21.17, 19.68],
                  [26.20,  25.00, 23.00]])
               
    cop = np.array([[2.53, 1.84, 4.41],
                    [2.76, 2.03, 1.56],
                    [3.10, 2.33, 1.81],
                    [3.90, 2.94, 2.31],
                    [4.60, 3.47, 2.70],
                    [4.90, 3.59, 2.93],
                    [5.19, 3.88, 3.01],
                    [5.95, 4.55, 3.41]])
               
    P = Q / cop
  
class LA60TU():
    t_ambient = np.array([-20, -15, -7, 2, 7, 10, 12, 20])
    t_flow = np.array([35, 45, 55])
    Q = np.array([[11.75,   12.56,  14.15], # Q(-20°C, W35) interpoliert
                  [15.39,   15.12,  13.22], # Q(-15°C, W35) interpoliert
                  [21.20,   17.84,  17.04],
                  [26.40,  23.90, 23.45],
                  [31.90,  23.79, 28.30],
                  [33.60,  32.50, 30.70],
                  [35.00,  34.60, 32.10],
                  [40.81,  40.11, 36.59]]) # Q(20°C, W35/45/55) interpoliert
               
    cop = np.array([[2.11, 1.72, 1.72], # COP(-20°C, W35) interpoliert
                    [2.45, 1.99, 1.57], # COP(-15°C, W35) interpoliert
                    [3.00, 2.28, 1.92],
                    [3.70, 2.81, 2.44],
                    [4.30, 3.34, 2.9],
                    [4.40, 3.65, 3.04],
                    [4.38, 3.84, 3.15],
                    [4.85, 4.20, 3.60]])
               
    P = Q / cop
    
def get_hp_data(Q_nom, t_ambient, t_flow):
    
    # Wie HP auswählen? Kleinste Differenz oder nächst größere/kleiner?
    # Was genau ist eigentlich Q_nom????
    
    hps = []
    hps.append(LA12TU())
    hps.append(LA9TU())
    hps.append(LA6TU())
    hps.append(LA17TU())
    hps.append(LA25TU())
    hps.append(LA40TU())
    hps.append(LA60TU())
    
    min_diff = 999999
    for data in hps:
	Q_nom_35 = 		basic_functions.heating_tech.interpolateQ(np.array([Constants.t_bivalent]), data.t_ambient, data.Q[:,0]) *1000
	Q_nom_55 = 		basic_functions.heating_tech.interpolateQ(np.array([Constants.t_bivalent]), data.t_ambient, data.Q[:,-1]) *1000
	bivalent_t_flow = 	basic_functions.heating_tech.heatingCurve(np.array([Constants.t_bivalent])+273.15) - 273.15
	diff = 			abs(basic_functions.heating_tech.heater_nominals(bivalent_t_flow, Q_nom_35, Q_nom_55) - Q_nom)
	
	if diff < min_diff:
	    min_diff = diff
	    hp = data

    Q_nom_35 = basic_functions.heating_tech.interpolateQ(t_ambient, hp.t_ambient, hp.Q[:,0]) *1000
    Q_nom_55 = basic_functions.heating_tech.interpolateQ(t_ambient, hp.t_ambient, hp.Q[:,-1]) *1000
    P_nom_35 = basic_functions.heating_tech.interpolateQ(t_ambient, hp.t_ambient, hp.P[:,0]) *1000
    P_nom_55 = basic_functions.heating_tech.interpolateQ(t_ambient, hp.t_ambient, hp.P[:,-1]) *1000
    
    return (basic_functions.heating_tech.heater_nominals(t_flow, Q_nom_35, Q_nom_55),
	    basic_functions.heating_tech.heater_nominals(t_flow, P_nom_35, P_nom_55))

	