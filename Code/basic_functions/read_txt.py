#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:45:32 2014

@author: tsz
"""

# Imports
import csv
import numpy as np

def read_single_indexed_file(path, delimiter = '\t'):
    """
    This function receives a filename and returns a list with the values of this file
    """
     
    with open(path, 'rb') as input:
        reader = csv.reader(input, delimiter=delimiter)
        result = []
        for row in reader:
            result.append(float(row[len(row)-1]))
            
    return np.array(result)
    
def read_multiple_indexed_file(path, delimiter = '\t'):
    """
    This function receives a filename and returns a list with the values of this file
    """
     
    with open(path, 'rb') as input:
        reader = csv.reader(input, delimiter=delimiter)
        result = []
        for row in reader:
            temp = []
            for i in xrange(0,len(row)):
                temp.append(float(row[i]))
                
            result.append(temp)
            
    return np.array(result)
   
def write_single_indexed_file(path, data, delimiter = '\t'):
    """
    This function writes the given data (array) into the specified path
    """
    
    length = len(data)
        
    with open(path, 'wb') as f:
        writer = csv.writer(f, delimiter = delimiter)
        for i in range(length):
            writer.writerow((i+1, data[i]))