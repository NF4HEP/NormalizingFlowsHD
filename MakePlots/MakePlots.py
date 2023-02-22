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
import Bijectors,Distributions,Metrics,MixtureDistributions,Plotters,Trainer,Utils


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



def cornerplotter(target_test_data,path_to_plots,ndims,rot=None,norm=False,max_dim=32):
    # Define the two samples (target and nf)
    shape = target_test_data.shape
    target_samples=target_test_data
    # Define generic labels
    labels = []
    for i in range(shape[1]):
        labels.append(r"$\theta_{%d}$" % i)
        i = i+1
    # Choose dimensions to plot
    thin = int(shape[1]/max_dim)+1
    if thin<=2:
        thin = 1
    # Select samples to plot
    target_samples = target_samples[:,::thin]

    # Select labels
    labels = list(np.array(labels)[::thin])

    red_bins=50
    density=(np.max(target_samples,axis=0)-np.min(target_samples,axis=0))/red_bins


    blue_line = mlines.Line2D([], [], color='red', label='target')
    red_line = mlines.Line2D([], [], color='blue', label='NF')
    figure=corner.corner(target_samples,color='red',bins=red_bins,labels=[r"%s" % s for s in labels])
    #corner.corner(nf_samples,color='blue',bins=blue_bins,fig=figure)
    #plt.legend(handles=[blue_line,red_line], bbox_to_anchor=(-ndims+1.8, ndims+.3, 1., 0.) ,fontsize='xx-large')
    plt.savefig(path_to_plots,pil_kwargs={'quality':50})
    plt.close()
    return



## Execute with
# nohup python "Main_CsplineN.py" 2> error_CsplineN.txt > output_CsplineN.txt &



def MixtureGaussian(ncomp,ndims,seed=0):
    targ_dist = MixtureDistributions.MixNormal1(ncomp,ndims,seed=seed)
    return targ_dist
def CorrMixtureGaussian(ncomp,ndims,seed=0):
    targ_dist = MixtureDistributions.MixMultiNormal1(ncomp,ndims,seed=seed)
    return targ_dist

### Initialize number of components ###

seed_dist = 0
seed_test = 0
ntest_samples=50000
ndims=4

ncomp=3
targ_dist=MixtureGaussian(ncomp,ndims,seed=seed_dist)
X_data_test=targ_dist.sample(ntest_samples,seed=seed_test).numpy()
path_to_umog='Umog_ref.pdf'
marginal_plot(X_data_test,path_to_umog,ndims)


ncomp=3
targ_dist=CorrMixtureGaussian(ncomp,ndims,seed=seed_dist)
X_data_test=targ_dist.sample(ntest_samples,seed=seed_test).numpy()
path_to_cmog='Cmog_ref.pdf'
cornerplotter(X_data_test,path_to_cmog,ndims)
   

ncomp=10
targ_dist=CorrMixtureGaussian(ncomp,ndims,seed=seed_dist)
X_data_test=targ_dist.sample(ntest_samples,seed=seed_test).numpy()
path_to_cmog_em='Cmogem_ref.pdf'
cornerplotter(X_data_test,path_to_cmog_em,ndims)
