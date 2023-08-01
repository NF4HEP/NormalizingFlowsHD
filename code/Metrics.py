

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:36:34 2019

@author: reyes-gonzalez
"""
from timeit import default_timer as timer
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
from Utils import reset_random_seeds
from typing import Optional, Tuple, List, Union

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
        ks_means.append(np.mean(pval_list))
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
    fn_list=[]
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
        fn_list.append(frob_norm)
    # Return the mean and std of the p-values
    return fn_list


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


def SWD(dist_1,
        dist_2,
        n_iter = 10,
        batch_size = 100000,
        n_slices = 100,
        seed = None):
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
    swd_lists=[]
    swd_means=[]
    swd_stds=[]
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
        swd_lists.append(swd_proj)
        # Compute the mean and std over iterations of the swd-values
        swd_means.append(np.mean(swd_proj))
        swd_stds.append(np.std(swd_proj))
    # Return the mean and std of the p-values
    return [swd_means,swd_stds,swd_lists]

@tf.function(reduce_retracing=True)
def wasserstein_distance_tf(dist1: tf.Tensor, 
                            dist2: tf.Tensor
                           ) -> tf.Tensor:
    # sort the distributions
    dist1_sorted = tf.sort(dist1, axis=-1)
    dist2_sorted = tf.sort(dist2, axis=-1)

    # calculate the differences between corresponding points in the sorted distributions
    diff = tf.abs(dist1_sorted - dist2_sorted)

    # calculate the mean of these differences
    emd = tf.reduce_mean(diff)
    
    return emd

@tf.function(reduce_retracing=True)
def swd_2samp_tf(dist_1: tf.Tensor, 
                 dist_2: tf.Tensor,
                 n_slices: int = 100,
                 ndims: int = 2
                ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    # Generate random directions
    directions = tf.random.normal((n_slices, ndims))
    directions /= tf.norm(directions, axis=1)[:, None]
    
    # Compute projections for all directions at once
    dist_1_proj = tf.tensordot(dist_1, directions, axes=[[1],[1]])
    dist_2_proj = tf.tensordot(dist_2, directions, axes=[[1],[1]])
    
    # Transpose the projection tensor to have slices on the first axis
    dist_1_proj = tf.transpose(dist_1_proj)
    dist_2_proj = tf.transpose(dist_2_proj)
    
    # Apply wasserstein_distance to each slice using tf.vectorized_map
    swd_proj = tf.vectorized_map(lambda args: wasserstein_distance_tf(*args), (dist_1_proj, dist_2_proj))
    
    # Compute mean and std
    swd_mean = tf.reduce_mean(swd_proj)
    swd_std = tf.math.reduce_std(swd_proj)

    return swd_mean, swd_std, swd_proj


def SWD_tf(dist_1: tf.Tensor,
           dist_2: tf.Tensor,
           n_iter: int = 10, 
           batch_size: int = 100000, 
           n_slices: int = 100, 
           dtype: tf.DType = tf.float32,
           seed: Optional[int] = None,
           max_vectorize: int = int(1e6),
           device: str = "/GPU:0",
           verbose: bool = False):
    
    with tf.device(device):
        # Prepare necessary variables
        max_vectorize = int(max_vectorize)
        dist_1_num: tf.Tensor = tf.cast(dist_1, dtype=tf.float32)
        dist_2_num: tf.Tensor = tf.cast(dist_2, dtype=tf.float32)
        nsamples, ndims = [int(i) for i in dist_1_num.shape]
        dtype: tf.DType = tf.as_dtype(dtype)
        start_time: float = 0.
        end_time: float = 0.
        elapsed: float = 0.
        if seed is None:
            seed = np.random.randint(1000000)
    
        # Utility functions
        @tf.function
        def conditional_tf_print(verbose: tf.Tensor = tf.convert_to_tensor(False), *args) -> None:
            tf.cond(tf.equal(verbose, True), lambda: tf.print(*args), lambda: verbose)
    
        def start_calculation() -> None:
            nonlocal start_time
            conditional_tf_print(verbose, "Starting KS tests calculation...")
            conditional_tf_print(verbose, "Running TF KS tests...")
            conditional_tf_print(verbose, "niter =", n_iter)
            conditional_tf_print(verbose, "batch_size =", batch_size)
            start_time = timer()
    
        def end_calculation() -> None:
            nonlocal end_time, elapsed
            end_time = timer()
            elapsed = end_time - start_time
            conditional_tf_print(verbose, "KS tests calculation completed in", str(elapsed), "seconds.")
    
        @tf.function(reduce_retracing=True)
        def compute_test() -> tf.Tensor:
            conditional_tf_print(verbose, "Running compute_test")
    
            # Initialize the result TensorArray
            res = tf.TensorArray(dtype, size=n_iter)
            res_swd_mean = tf.TensorArray(dtype, size=n_iter)
            res_swd_std = tf.TensorArray(dtype, size=n_iter)
            res_swd_proj = tf.TensorArray(dtype, size=n_iter)
    
            def body(i, res):
                # Define the loop body to vectorize over ndims*chunk_size
                dist_1_k: tf.Tensor = dist_1_num[i * batch_size: (i + 1) * batch_size, :]
                dist_2_k: tf.Tensor = dist_2_num[i * batch_size: (i + 1) * batch_size, :]
        
                swd_mean, swd_std, swd_proj = swd_2samp_tf(dist_1_k, dist_2_k, n_slices = n_slices, ndims = ndims)
                swd_mean = tf.cast(swd_mean, dtype=dtype)
                swd_std = tf.cast(swd_std, dtype=dtype)
                swd_proj = tf.cast(swd_proj, dtype=dtype)
        
                # Here we add an extra dimension to `swd_mean` and `swd_std` tensors to match rank with `swd_proj`
                swd_mean = tf.expand_dims(swd_mean, axis=0)
                swd_std = tf.expand_dims(swd_std, axis=0)
        
                result_value = tf.concat([swd_mean, swd_std, swd_proj], axis=0)
        
                res = res.write(i, result_value)
                return i+1, res
    
            _, res = tf.while_loop(lambda i, _: i < n_iter, body, [0, res])
            
            for i in range(n_iter):
                res_i = res.read(i)
                res_swd_mean = res_swd_mean.write(i,res_i[0])
                res_swd_std = res_swd_std.write(i,res_i[1])
                res_swd_proj = res_swd_proj.write(i,res_i[2:])
            
            swd_means = res_swd_mean.stack()
            swd_stds = res_swd_std.stack()
            swd_lists = res_swd_proj.stack()
                            
            return [swd_means, swd_stds, swd_lists]
    
        start_calculation()
    
        reset_random_seeds(seed=seed)
    
        result_value: tf.Tensor = compute_test()
        
        result_value = [i.numpy().tolist() for i in result_value]
    
        end_calculation()
    
        return result_value


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


def ComputeMetrics(dist_1,dist_2,n_iter=1,batch_size=100000,n_slices=100,seed=None,verbose=True):
    """
    Function that computes the metrics. The following metrics are implemented:
    
        - Mean and median of 1D KS-test
        - Mean and median of 1D Anderson-Darling test
        - Frobenius norm
        - Mean and median of Wasserstein distance
        - Sliced Wasserstein distance
    """
    def conditional_print(*args):
        if verbose:
            print(args)
    conditional_print("Computing KS test")
    start = timer()
    [ks_means, ks_stds, ks_lists] = KS_test(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size)
    end = timer()
    conditional_print("Time elapsed: ", end-start)
    conditional_print("Computing AD test")
    start = timer()
    [ad_means, ad_stds, ad_lists] = AD_test(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size)
    end = timer()
    conditional_print("Time elapsed: ", end-start)
    conditional_print("Computing FN test")
    start = timer()
    fn_list = FN(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size)
    end = timer()
    conditional_print("Time elapsed: ", end-start)
    conditional_print("Computing WD test")
    start = timer()
    [wd_means, wd_stds, wd_lists] = WD(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size)
    end = timer()
    conditional_print("Time elapsed: ", end-start)
    conditional_print("Computing SWD test")
    start = timer()
    [swd_means, swd_stds, swd_lists] = SWD(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size, n_slices = n_slices, seed = seed)
    end = timer()
    conditional_print("Time elapsed: ", end-start)
    return ks_means, ks_stds, ks_lists, ad_means, ad_stds, ad_lists, fn_list, wd_means, wd_stds, wd_lists, swd_means, swd_stds, swd_lists


def ComputeMetricsReduced(dist_1,dist_2,n_iter=1,batch_size=100000,n_slices=100,seed=None,verbose=True):
    """
    Function that computes the metrics. The following metrics are implemented:
    
        - Mean and median of 1D KS-test
        - Frobenius norm
        - Sliced Wasserstein distance
    """
    def conditional_print(*args):
        if verbose:
            print(*args)
    conditional_print("Computing KS test")
    start = timer()
    [ks_means, ks_stds, ks_lists] = KS_test(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size)
    end = timer()
    conditional_print("KS test computed in ", end-start)
    conditional_print("Computing FN values")
    start = timer()
    fn_list = FN(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size)
    end = timer()
    conditional_print("FN computed in ", end-start)
    conditional_print("Computing SWD values")
    start = timer()
    [swd_means, swd_stds, swd_lists] = SWD(dist_1, dist_2, n_iter = n_iter, batch_size = batch_size, n_slices = n_slices, seed = seed)
    end = timer()
    conditional_print("SWD computed in ", end-start)
    return ks_means, ks_stds, ks_lists, [], [], [], fn_list, [], [], [], swd_means, swd_stds, swd_lists


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
