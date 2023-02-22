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

def load_sample(path_to_results):

    nf_sample=np.load(path_to_results+'/nf_sample.npy',allow_pickle=True)
    #nf_sample=np.load(path_to_results+'/sample_nf.pcl',allow_pickle=True)
    return nf_sample

def marginal_plot(target_test_data,nf_sample_maf,mf_sample_arqs,path_to_plots,ndims):

 
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

            axs[row,column].hist(target_test_data[:,dim], bins=n_bins,density=True,histtype='step',color='red',label='True')
            axs[row,column].hist(nf_sample_maf[:,dim], bins=n_bins,density=True,histtype='step',color='blue',label='MAF')
            axs[row,column].hist(nf_sample_arqs[:,dim], bins=n_bins,density=True,histtype='step',color='green',label='A-RQS')
            
        
            x_axis = axs[row,column].axes.get_xaxis()
            x_axis.set_visible(False)
            y_axis = axs[row,column].axes.get_yaxis()
            y_axis.set_visible(False)
            if dim==11:
                axs[row,column].legend(loc='upper right')
        
    fig.savefig(path_to_plots,dpi=300)
    fig.clf()

    return


ndims=16
seed_test=0
ntest_samples=100000
targ_dist=TruncatedGaussian(ndims)
X_data_test=targ_dist.sample(ntest_samples,seed=seed_test).numpy()


path_to_maf='/Users/humberto/Documents/work/NFs/github-NFs2/NormalizingFlowsHD-2/Riccardo/Truncated/results/BestsForMarcoTruncated16/MAF/'
nf_sample_maf=load_sample(path_to_maf)


path_to_arqs='/Users/humberto/Documents/work/NFs/github-NFs2/NormalizingFlowsHD-2/Riccardo/Truncated/results/BestsForMarcoTruncated16/Mspline/'
nf_sample_arqs=load_sample(path_to_arqs)


path_to_plots='ensemple_truncated.pdf'
marginal_plot(X_data_test,nf_sample_maf,nf_sample_arqs,path_to_plots,ndims)
