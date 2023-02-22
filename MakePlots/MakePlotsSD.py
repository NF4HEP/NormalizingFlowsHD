import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb = tfp.bijectors
import pandas as pd
import pickle
from timeit import default_timer as timer
import traceback
from typing import Dict, Any
import matplotlib.lines as mlines
import corner
import matplotlib.pyplot as plt




sys.path.append('../code')
import Bijectors,Distributions,Metrics,MixtureDistributions,Plotters,Trainer,Utils,SpecialDistributions



def TruncatedGaussian(ndims):

    targ_dist = SpecialDistributions.TruncatedDistributions(ndims)
    
    return targ_dist
def AllResonance(ndims):

    targ_dist = SpecialDistributions.AllResonanceDistributions(ndims)
    
    return targ_dist

def marginal_plot(target_test_data,path_to_plots,ndims):

 
    n_bins=50

    

    if ndims<=4:
    
        fig, axs = plt.subplots(int(ndims/2), 2, tight_layout=True)
        for dim in range(ndims):
    
            row=int(dim/2)
            column=int(dim%2)

            axs[row,column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')

        
        
            x_axis = axs[row,column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[row,column].axes.get_yaxis()
            y_axis.set_visible(False)
    
    
    

    elif ndims>=100:
    
        fig, axs = plt.subplots(int(ndims/10), 10, tight_layout=True)
    
        for dim in range(ndims):
    
  
            row=int(dim/10)
            column=int(dim%10)

            axs[row,column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            #axs[row,column].hist(sample_nf[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
        
        
            x_axis = axs[row,column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[row,column].axes.get_yaxis()
            y_axis.set_visible(False)

    else:
        
        
        fig, axs = plt.subplots(int(ndims/4), 4, tight_layout=True)
        for dim in range(ndims):
    
            row=int(dim/4)
            column=int(dim%4)

            axs[row,column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red')
            #axs[row,column].hist(sample_nf[:,dim], bins=n_bins,density=True,histtype='step',color='blue')
        
        
            x_axis = axs[row,column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[row,column].axes.get_yaxis()
            y_axis.set_visible(False)
        
    fig.savefig(path_to_plots,dpi=300)
    fig.clf()

    return



seed_dist = 0
seed_test = 0
ntest_samples=50000
ndims=8

targ_dist=TruncatedGaussian(ndims)
X_data_test=targ_dist.sample(ntest_samples,seed=seed_test).numpy()
path_to_trunc='trunc_ref.pdf'
marginal_plot(X_data_test,path_to_trunc,ndims)


targ_dist=AllResonance(ndims)
X_data_test=targ_dist.sample(ntest_samples,seed=seed_test).numpy()
path_to_reson='reson_ref.pdf'
marginal_plot(X_data_test,path_to_reson,ndims)
