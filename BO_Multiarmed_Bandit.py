#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:54:09 2020

@author: marc
"""

"""
This is the code used for the multiarmed bandit optimization problem. The goal is to use the GPyOpt package
to optimize the IC50 of MBL inhibitors obtained from the Schofield dataset.

-EDITED 9.08.2020 by Marc Moesser to fix the kernel issue and generalize it to use one script for all proteins and features.

"""


#For Bayesian optimization
import GPyOpt
import GPy
import numpy as np
from numpy.random import seed
import pandas as pd
import sys
#My own packages




""" UTILITY FUNCTIONS FOR THE BETTER PAIRING OF FINGERPRINTS AND pIC50 VALUES"""

#This function takes in an ECFP, CSFP or mol2vec and outputs it as a string to be used as a key in a dictionary
def FP_to_bitstr(FP):
    return ''.join([str(b)[0] for b in FP.tolist()])
#This function uses the desired fingerprint and the dict where it is used as a key and returns the corresponding value in the dict
def convert_FP_to_IC50(FP):
    FP_1D = FP[0]
    return np.asarray([[-FP_to_pIC50[FP_to_bitstr(FP_1D)]]])

#This function uses the desired mol2vector and the dict where it is used as a key and returns the corresponding value in the dict
#ATTENTION: The bayesian optimization algo adds a [] to the string which means we have to remove it again for the dict search to be accurate
#Otherwise it will search for [key] instead of key -> which will result in a KeyError since that key doesnt exist.
def convert_mol2vec_to_IC50(mol2vec):
    mol2vec_2dim = mol2vec[0]
    x = FP_to_pIC50[FP_to_bitstr(mol2vec_2dim)]
    return np.asarray([[-x]])


 
""" =========================================================================================================

                                        THIS IS THE MAIN PART OF THE CODE

=========================================================================================================="""

#Set all the hyperparameters for the script 
protein_target, feature, kernel = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

# the options are:
#protein_target = "string name for your dataset"
#feature = ["mol2vec", "CSFP", "ECFP"]
#kernel = ["rbfs", "tanimoto"]

file_location = "./input_data/"+protein_target+"_"+feature+".csv"

n_seedmol = 3
print("The protein target is: ", protein_target, "The representation is: ", feature, "The kernel is: ", kernel)

#Load in dataframe
df_clean = pd.read_csv(file_location)

#Get the IC50 values
pIC50_np = np.asarray(df_clean["pIC50"].tolist())
#Get the fingerprints into an array
FP_np = []
for index in range(0, len(df_clean)):
    individual_fp = np.asarray([int(x) for x in df_clean.iloc[index][1:-1]])
    FP_np.append(individual_fp)
FP_np = np.asarray(FP_np)

#This turns the FPs into a string and makes a list of all strings. They will be used as keys in a dictionary for sampling
list_of_keys = [FP_to_bitstr(i)  for i in FP_np]

#This function maps IC50 and FP into a dictionary
FP_to_pIC50 = {key:value for (key,value) in zip(list_of_keys, pIC50_np)}

domain = [{"name": "inhibitors", "type": "bandit", "domain":FP_np}]

# Set the kernel
if kernel == "rbfs":
    kernel_mod = GPy.kern.RBF(input_dim=FP_np.shape[1], variance = 1, lengthscale=5)
if kernel == "tanimoto":
    kernel_mod = GPy.kern.Tanimoto(input_dim=FP_np.shape[1])

#Set the obj function
if feature == "mol2vec":
    obj_function = convert_mol2vec_to_IC50
if feature == "CSFP" or feature == "ECFP":
    obj_function = convert_FP_to_IC50
    
se = 1
max_iter = 200 

for x in range(10):
    seed(se)
    model = GPyOpt.models.GPModel(kernel=kernel_mod, exact_feval=True,optimize_restarts=10,verbose= False) 
    objective = GPyOpt.core.task.SingleObjective(obj_function)
    space = GPyOpt.Design_space(domain)
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
    initial_design = GPyOpt.experiment_design.initial_design('random', space, n_seedmol)
    acquisition_EI = GPyOpt.acquisitions.AcquisitionEI(model, space, optimizer=aquisition_optimizer)
    evaluator_EI = GPyOpt.core.evaluators.Sequential(acquisition_EI)
    EI_bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition_EI, evaluator_EI, initial_design, normalize_Y=False, de_duplication=True)
    EI_bo.run_optimization( max_iter = max_iter, evaluations_file="./run_output/"+protein_target+"_"+feature+"_"+kernel+"_seed%i.txt" %se)#
    
    del EI_bo # delete objective
    print("seed number: ", se, " done, starting next seed")
    se += 1

print("done")

