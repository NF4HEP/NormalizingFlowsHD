

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:36:34 2019

@author: reyes-gonzalez
"""
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.stats import epps_singleton_2samp
from scipy.stats import anderson_ksamp
from statistics import mean,median


def KL_divergence(target_test_data,nf_dist,test_log_prob):

    #q_sampler = targ_dist.sample(nsamples)

    #q_density = targ_dist.log_prob(target_test_data)
    p_density=nf_dist.log_prob(target_test_data)
    KL_estimate = np.mean(test_log_prob - p_density)
    
    return KL_estimate
    
def Wasserstein_distance(target_test_data,nf_dist,norm=True):

    ##create data sample from trained normising flow
    #z=base_dist.sample((target_test_data.shape[0]))
    #x_estimated=nf_dist.bijector.forward(z).numpy()
    if norm==False:
        x_estimated=nf_dist.sample(target_test_data.shape[0])
        x_estimated=np.reshape(x_estimated,newshape=target_test_data.shape)
    else:
        x_estimated=nf_dist
    wasserstein_distances=[]
    for dim in range(target_test_data.shape[1]):
        #print(wasserstein_distance(x_target[:,dim], x_estimated[:,dim]))
        ws_distance=wasserstein_distance(target_test_data[:,dim], x_estimated[:,dim])
        wasserstein_distances.append(ws_distance)
        
    return wasserstein_distances

def sliced_Wasserstein_distance(target_test_data, nf_dist, norm=True, n_slices=None, seed=None):
    """
    Compute the sliced Wasserstein distance between two sets of points
    using n_slices random directions and the p-th Wasserstein distance.
    """
    if seed is None:
        np.random.seed(np.random.randint(10e6))
    else:
        np.random.seed(int(seed))
    if n_slices is None:
        n_slices = target_test_data.shape[1]
    else:
        n_slices = int(n_slices)
    if norm==False:
        x_estimated=nf_dist.sample(target_test_data.shape[0])
        x_estimated=np.reshape(x_estimated,newshape=target_test_data.shape)
    else:
        x_estimated=nf_dist
    # Generate random directions
    directions = np.random.randn(n_slices, target_test_data.shape[1])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    # Compute sliced Wasserstein distance
    ws_distances = []
    for direction in directions:
        target_proj = target_test_data @ direction
        estimated_proj = x_estimated @ direction
        ws_distances.append(wasserstein_distance(target_proj, estimated_proj))
    mean = np.mean(ws_distances)
    std = np.std(ws_distances)
    return [mean,std]

def KS_test(target_test_data,nf_dist,n_iter=100,norm=True):
    
    ndims=target_test_data.shape[1]
    nsamples=target_test_data.shape[0]
    batch_size=int(nsamples/n_iter)
    
    
    ###### Now we compute ks test between two different dists and print out the norm
    big_list=[]
    for dim in range(ndims):
        dim_list=[]
        big_list.append(dim_list)
    
    for k in range(n_iter):
        ## create new sample from target distribution
        batch_test=target_test_data[k*batch_size:(k+1)*batch_size,:]
        ##create data sample from trained normising flow
        #z=base_dist.sample((nsamples))
        #x_estimated=nf_dist.bijector.forward(z).numpy()
        if norm==False:
            x_estimated=nf_dist.sample(batch_test.shape[0])
            x_estimated=np.reshape(x_estimated,newshape=batch_test.shape)
        else:
            x_estimated=nf_dist[k*batch_size:(k+1)*batch_size,:]
        
        for dim in range(ndims):
            p_val=stats.ks_2samp(x_estimated[:,dim], batch_test[:,dim])[1]
            big_list[dim].append(p_val)
    
    ks_test_all=[]
    for dim in range(ndims):
        
        ks_test_dim=float(np.mean(big_list[dim]))
        ks_test_all.append(ks_test_dim)
        
    return ks_test_all
    

def correlation_from_covariance(covariance):
   
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    

    return correlation

def FrobNorm(target_test_data,nf_dist,norm=True):


    #covariance target distribtuion
    target_cov=np.cov(target_test_data,bias=True,rowvar=False)
    target_cov=np.tril(target_cov)

    #covariance nf
    
    if norm==False:
        nf_sample=nf_dist.sample(target_test_data.shape[0])
    else:
        nf_sample=nf_dist
    

    #nf_sample=np.reshape(nf_sample,newshape=(nsamples_frob_norm, ndims))
    nf_cov = np.cov(nf_sample,bias=True,rowvar=False)
    nf_cov=np.tril(nf_cov)

    #correlation matrices
    nf_corr=correlation_from_covariance(nf_cov)
    target_corr=correlation_from_covariance(target_cov)

    #frobenius norm of correlation matrices
    matrix_sum=nf_corr-target_corr
    frob_norm=np.linalg.norm(matrix_sum, ord='fro')

    return frob_norm,nf_corr,target_corr

def AD_test(target_test_data,nf_dist,n_iter=100,norm=True):

    ndims=target_test_data.shape[1]
    nsamples=target_test_data.shape[0]
    batch_size=int(nsamples/n_iter)

    ###### Now we compute ks test between tow different dists and print out the norm
    big_list=[]
    for dim in range(ndims):
        dim_list=[]
        big_list.append(dim_list)

    for k in range(n_iter):
        ## create new sample from target distribution
        batch_test=target_test_data[k*batch_size:(k+1)*batch_size,:]
        ##create data sample from trained normising flow
        #z=base_dist.sample((nsamples))
        #x_estimated=nf_dist.bijector.forward(z).numpy()
        if norm==False:
            x_estimated=nf_dist.sample(batch_test.shape[0])
            x_estimated=np.reshape(x_estimated,newshape=batch_test.shape)

        else:
            x_estimated=nf_dist[k*batch_size:(k+1)*batch_size,:]


        for dim in range(ndims):
            p_val=anderson_ksamp([x_estimated[:,dim], batch_test[:,dim]])[2]

            big_list[dim].append(p_val)
            
    AD_test_all=[]
    for dim in range(ndims):
        AD_norm=float(np.mean(big_list[dim]))
        AD_test_all.append(AD_norm)

    return AD_test_all

def ComputeMetrics(X_data_test,nf_dist):
    """
    Function that computes the metrics. The following metrics are implemented:
    
        - KL-divergence
        - Mean and median of 1D KS-test
        - Mean and median of 1D Anderson-Darling test
        - Mean and median of Wasserstein distance
        - Frobenius norm
    """
    kl_divergence=-1
    ks_test_list=KS_test(X_data_test,nf_dist)
    ks_median=median(ks_test_list)
    ks_mean=mean(ks_test_list)
    ad_test_list=AD_test(X_data_test,nf_dist)
    ad_median=median(ad_test_list)
    ad_mean=mean(ad_test_list)
    w_distance_list=Wasserstein_distance(X_data_test,nf_dist)
    w_distance_median=median(w_distance_list)
    w_distance_mean=median(w_distance_list)
    frob_norm,nf_corr,target_corr=FrobNorm(X_data_test,nf_dist)
    return kl_divergence,ks_median,ks_mean,ad_median,ad_mean,w_distance_median,w_distance_mean,frob_norm,nf_corr,target_corr

'''
def JS_distance(target_test_data,nf_dist,test_log_prob):

    #q_sampler = targ_dist.sample(nsamples)
    q_density = targ_dist.log_prob(target_test_data)
    p_density=nf_dist.log_prob(target_test_data)
    KL_estimate_qp = np.mean(q_density - p_density)
    
    nsamples=target_test_data.shape[0]
    p_sampler = nf_dist.sample(nsamples)
    p_density = nf_dist.log_prob(p_sampler)
    q_density=targ_dist.log_prob(p_sampler)
    KL_estimate_pq = np.mean(p_density - q_density)
    
    js_distance=np.sqrt((KL_estimate_qp+KL_estimate_pq)/2)
    
    return js_distance

'''

'''

def ES_test(target_test_data,nf_dist,base_dist,n_iter=100):
    
    ndims=target_test_data.shape[1]
    nsamples=target_test_data.shape[0]
    batch_size=int(nsamples/n_iter)

    
    ###### Now we compute ks test between tow different dists and print out the norm
    big_list=[]
    for dim in range(ndims):
        dim_list=[]
        
        big_list.append(dim_list)
    
    for k in range(n_iter):
        ## create new sample from target distribution
        batch_test=target_test_data[k*batch_size:(k+1)*batch_size,:]
        ##create data sample from trained normising flow
        #z=base_dist.sample((nsamples))
        #x_estimated=nf_dist.bijector.forward(z).numpy()
        x_estimated=nf_dist.sample(batch_test.shape[0])
        x_estimated=np.reshape(x_estimated,newshape=batch_test.shape)
        
        for dim in range(ndims):
            p_val=epps_singleton_2samp(x_estimated[:,dim], batch_test[:,dim])[1]
            big_list[dim].append(p_val)
    
    es_test_all=[]
    for dim in range(ndims):
        es_norm=float(np.mean(big_list[dim]))
        es_test_all.append(es_norm)
        
    return es_test_all

'''
