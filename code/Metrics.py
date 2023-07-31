

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

def correlation_from_covariance(covariance):
    """Computes the correlation matrix from the covariance matrix.

    Args:
        covariance (array): Covariance matrix

    Returns:
        _type_: array
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def KL_divergence(target_test_data,nf_dist,test_log_prob):
    """Computes the KL divergence between the target distribution and the NF distribution.

    Args:
        target_test_data (_type_): _description_
        nf_dist (_type_): _description_
        test_log_prob (_type_): _description_

    Returns:
        _type_: _description_
    """
    p_density=nf_dist.log_prob(target_test_data)
    KL_estimate = np.mean(test_log_prob - p_density)
    return KL_estimate

def KS_test(dist_1,dist_2,n_iter=10,batch_size=100000):
    """
    The Kolmogorov-Smirnov test is a non-parametric test that compares two distributions and returns a p-value that indicates whether the two distributions are the same or not. 
    The test is performed for each dimension of the distributions and for n_iter times and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in n_iter batches dist_1_j, dist_2_j of size batch_size=int(nsamples/n_iter) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_j.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 10.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """    
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    # Define ks_lists, ks_means and ks_stds that will contain the list of p-values over dimensions, mean of p-values over dimensions and std of p-values over dimensions for all iterations
    ks_lists=[]
    ks_means=[]
    ks_stds=[]
    # Loop over all iterations
    for k in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_k=dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfd.distribution.Distribution):
            dist_1_k=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        if isinstance(dist_2, np.ndarray):
            dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfd.distribution.Distribution):
            dist_2_k=dist_2.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        # The ks test is computed and the p-value saved for each dimension
        pval_list = []
        for dim in range(ndims):
            p_val=stats.ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim])[1]
            pval_list.append(p_val)
        # Save the list of p-values for each iteration
        ks_lists.append(pval_list)
        # Compute the mean and std over dimensions of the p-values for each iteration
        ks_means.append(np,mean(pval_list))
        ks_stds.append(np.std(pval_list))
    # Return the mean and std of the p-values
    return [ks_means,ks_stds,ks_lists]

def KS_test_old(target_test_data,nf_dist,n_iter=100,norm=True):
    """Kolmogorov-Smirnov test between target distribution and trained normalising flow 
    Args:
        target_test_data (_type_): _description_
        nf_dist (_type_): _description_
        n_iter (int, optional): _description_. Defaults to 100.
        norm (bool, optional): _description_. Defaults to True.
    Returns:
        _type_: _description_
    """
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


def AD_test(dist_1,dist_2,n_iter=10,batch_size=100000):
    """
    The Anderson-Darling test is a non-parametric test that compares two distributions and returns a p-value that indicates whether the two distributions are the same or not. 
    The test is performed for each dimension of the distributions and for n_iter times and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in n_iter batches dist_1_j, dist_2_j of size batch_size=int(nsamples/n_iter) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_j.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 10.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    # Define ad_list, ad_mean and ad_std that will contain the list of p-values over dimensions, mean of p-values over dimensions and std of p-values over dimensions for all iterations
    ad_lists=[]
    ad_means=[]
    ad_stds=[]
    # Loop over all iterations
    for k in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_k=dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfd.distribution.Distribution):
            dist_1_k=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        if isinstance(dist_2, np.ndarray):
            dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfd.distribution.Distribution):
            dist_2_k=dist_2.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        # The ad test is computed and the p-value saved for each dimension
        pval_list = []
        for dim in range(ndims):
            p_val=anderson_ksamp([dist_1_k[:,dim], dist_2_k[:,dim]])[2]
            pval_list.append(p_val)
        # Save the list of p-values for each iteration
        ad_lists.append(pval_list)
        # Compute the mean and std over dimensions of the p-values for each iteration
        ad_means.append(np.mean(pval_list))
        ad_stds.append(np.std(pval_list))
    # Return the mean and std of the p-values
    return [ad_means,ad_stds,ad_lists]

def AD_test_old(target_test_data,nf_dist,n_iter=100,norm=True):
    """Anderson-Darling test between target distribution and trained normalising flow
    Args:
        target_test_data (_type_): _description_
        nf_dist (_type_): _description_
        n_iter (int, optional): _description_. Defaults to 100.
        norm (bool, optional): _description_. Defaults to True.
    Returns:
        _type_: _description_
    """
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


def FN(dist_1,dist_2,n_iter=10,batch_size=100000):
    """
    The Frobenius-Norm of the difference between the correlation matrices of two distributions.
    The norm is computed for n_iter times and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in n_iter batches dist_1_j, dist_2_j of size batch_size=int(nsamples/n_iter) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_j.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 10.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    # Define fn_list that will contain the list of fn for all dimensions and all iterations
    FN_list=[]
    # Loop over all iterations
    for k in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_k=dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfd.distribution.Distribution):
            dist_1_k=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        if isinstance(dist_2, np.ndarray):
            dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfd.distribution.Distribution):
            dist_2_k=dist_2.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        # The fn test is computed and the p-value saved for each dimension
        dist_1_cov = np.cov(dist_1_k,bias=True,rowvar=False)
        dist_1_corr=correlation_from_covariance(dist_1_cov)
        dist_2_cov = np.cov(dist_2_k,bias=True,rowvar=False)
        dist_2_corr=correlation_from_covariance(dist_2_cov)    
        matrix_sum=dist_1_corr-dist_2_corr
        frob_norm=np.linalg.norm(matrix_sum, ord='fro')
        FN_list.append(frob_norm)
    # Compute the mean and std of the p-values
    FN_mean = np.mean(FN_list)
    FN_std = np.std(FN_list)
    # Return the mean and std of the p-values
    return [FN_mean,FN_std, FN_list]


def FrobNorm_old(target_test_data,nf_dist,norm=True):
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


def WD(dist_1,dist_2,n_iter=10,batch_size=100000):
    """
    The Wasserstein distance between the target distribution and the distribution of the test data.
    The distance is computed for n_iter times and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in n_iter batches dist_1_j, dist_2_j of size batch_size=int(nsamples/n_iter) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_j.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 10.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.

    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    # Define ad_list that will contain the list of wd for all dimensions and all iterations
    wd_lists=[]
    wd_means=[]
    wd_stds=[]
    # Loop over all iterations
    for k in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_k=dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfd.distribution.Distribution):
            dist_1_k=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        if isinstance(dist_2, np.ndarray):
            dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfd.distribution.Distribution):
            dist_2_k=dist_2.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        # The WD test is computed and saved for each dimension
        wd_dims=[]
        for dim in range(ndims):
            wd=wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim])
            wd_dims.append(wd)
        # Save the list of wd-values for each iteration
        wd_lists.append(wd_dims)
        # Compute the mean and std over dimensions of the wd-values for each iteration
        wd_means.append(np.mean(wd_dims))
        wd_stds.append(np.std(wd_dims))
    # Return the mean and std of the p-values
    return [wd_means,wd_stds, wd_lists]


def Wasserstein_distance_old(target_test_data,nf_dist,norm=True):

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


def SWD(dist_1,dist_2,n_iter=10,batch_size=100000,n_slices=100,seed=None):
    """
    Compute the sliced Wasserstein distance between two sets of points using n_slices random directions and the p-th Wasserstein distance.
    The distance is computed for n_iter times and for n_slices directions and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in n_iter batches dist_1_j, dist_2_j of size batch_size=int(nsamples/n_iter) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_j.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        n_iter (int, optional): Number of iterations to be performed. Defaults to 100.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/n_iter
    if isinstance(dist_1, np.ndarray):
        ndims=dist_1.shape[1]
        nsamples=dist_1.shape[0]
        batch_size=int(nsamples/n_iter)
    elif isinstance(dist_2, np.ndarray):
        ndims=dist_2.shape[1]
        nsamples=dist_2.shape[0]
        batch_size=int(nsamples/n_iter)
    else:
        ndims=dist_1.sample(2).numpy().shape[1]
    if seed is None:
        np.random.seed(np.random.randint(1000000))
    else:
        np.random.seed(int(seed))
    if n_slices is None:
        n_slices = np.max([100,ndims])
    else:
        n_slices = int(n_slices)
    # Define ad_list that will contain the list of swd for all dimensions and all iterations
    swd_list=[]
    swd_mean=[]
    swd_std=[]
    # Loop over all iterations
    for k in range(n_iter):
        # If num is true, then the samples are split in n_iter batches of size nsamples/n_iter, otherwise we just sample batch_size points from the distributions
        if isinstance(dist_1, np.ndarray):
            dist_1_k=dist_1[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_1, tfd.distribution.Distribution):
            dist_1_k=dist_1.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        if isinstance(dist_2, np.ndarray):
            dist_2_k=dist_2[k*batch_size:(k+1)*batch_size,:]
        elif isinstance(dist_2, tfd.distribution.Distribution):
            dist_2_k=dist_2.sample(batch_size).numpy()
        else:   
            raise ValueError("dist_1 must be either a numpy array or a distribution")
        # Generate random directions
        directions = np.random.randn(n_slices, ndims)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        # Compute sliced Wasserstein distance
        swd_proj = []
        for direction in directions:
            dist_1_proj = dist_1_k @ direction
            dist_2_proj = dist_2_k @ direction
            swd_proj.append(wasserstein_distance(dist_1_proj, dist_2_proj))
        # Save the swd-value for each iteration
        swd_list.append(np.mean(swd_proj))
    # Compute the mean and std over iterations of the swd-values
    swd_mean.append(np.mean(swd_list))
    swd_std.append(np.std(swd_list))
    # Return the mean and std of the p-values
    return [swd_mean,swd_std,swd_list]


def sliced_Wasserstein_distance_old(target_test_data, nf_dist, norm=True, n_slices=None, seed=None):
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


def ComputeMetrics(dist_1,dist_2,n_iter=10,batch_size=100000,n_slices=100,seed=None):
    """
    Function that computes the metrics. The following metrics are implemented:
    
        - KL-divergence
        - Mean and median of 1D KS-test
        - Mean and median of 1D Anderson-Darling test
        - Mean and median of Wasserstein distance
        - Frobenius norm
    """
    [ks_mean,ks_std,ks_list]=KS_test(dist_1,dist_2,n_iter=n_iter,batch_size=batch_size)
    [ad_mean,ad_std,ad_list]=AD_test(dist_1,dist_2,n_iter=n_iter,batch_size=batch_size)
    [fn_mean,fn_std,fn_list]=FN(dist_1,dist_2,n_iter=n_iter,batch_size=batch_size)
    [wd_mean,wd_std,wd_list]=WD(dist_1,dist_2,n_iter=n_iter,batch_size=batch_size)
    [swd_mean,swd_std,swd_list]=SWD(dist_1,dist_2,n_iter=n_iter,batch_size=batch_size,n_slices=n_slices,seed=seed)
    return ks_mean,ks_std,ks_list,ad_mean,ad_std,ad_list,wd_mean,wd_std,wd_list,swd_mean,swd_std,swd_list,fn_mean,fn_std,fn_list


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
