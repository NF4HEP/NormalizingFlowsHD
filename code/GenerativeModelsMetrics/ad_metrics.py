#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated June 2023

@author: Riccardo Torre (riccardo.torre@ge.infn.it)
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import traceback
from timeit import default_timer as timer
from tqdm import tqdm # type: ignore
from scipy import stats # type: ignore
from scipy.stats import ks_2samp # type: ignore # Need a tf version
from scipy.stats import anderson_ksamp # type: ignore # Need a tf version
from scipy.stats import wasserstein_distance # type: ignore # Need a tf version
from statistics import mean,median
from typing import List, Tuple, Dict, Callable, Union, Optional

def ks_2samp_wrapper(dist_1, dist_2):
    return ks_2samp(dist_1, dist_2)

def anderson_ksamp_wrapper(dist_1, dist_2):
    return anderson_ksamp(dist_1, dist_2)

def wasserstein_distance_wrapper(dist_1, dist_2):
    return wasserstein_distance(dist_1, dist_2)
    
    
def __get_best_dtype(dtype1, dtype2):
    dtype1_precision = tf.as_dtype(dtype1).min
    dtype2_precision = tf.as_dtype(dtype2).min
    return tf.cond(dtype1_precision > dtype2_precision, lambda: dtype1, lambda: dtype2)

@tf.function
def conditional_tf_print(string: str,
                         verbose: bool = False) -> None:
    tf.cond(tf.equal(verbose, True), lambda: tf.print(string), lambda: tf.constant(verbose))

def correlation_from_covariance_np(covariance: np.ndarray) -> np.ndarray:
    """
    Function that computes the correlation matrix from the covariance matrix.
    
    Args:
        covariance: np.ndarray or tf.Tensor, covariance matrix.
        
    Returns:
        np.ndarray or tf.Tensor, correlation matrix.
    """
    stddev: np.ndarray = np.sqrt(np.diag(covariance))
    correlation: np.ndarray = covariance / np.outer(stddev, stddev)
    correlation = np.where(np.equal(covariance, 0), 0, correlation) 
    return correlation


def correlation_from_covariance_tf(covariance: tf.Tensor) -> tf.Tensor:
    """
    Function that computes the correlation matrix from the covariance matrix.
    
    Args:
        covariance: np.ndarray or tf.Tensor, covariance matrix.
        
    Returns:
        np.ndarray or tf.Tensor, correlation matrix.
    """
    stddev = tf.sqrt(tf.linalg.diag_part(covariance))
    correlation = covariance / tf.tensordot(stddev[:, None], stddev[None, :], axes=0)
    correlation = tf.where(tf.equal(covariance, 0), 0, correlation) 
    return correlation


def __parse_input_dist(dist_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                     verbose: bool = False
                    ) -> Tuple[Optional[tfp.distributions.Distribution], tf.Tensor, int, Optional[int], tf.DType]:
    dist_symb: tfp.distributions.Distribution
    dist_num: tf.Tensor
    nsamples: Optional[int]
    ndims: int
    if verbose:
        print("Parsing input distribution...")
    if isinstance(dist_input, (np.ndarray, tf.Tensor)):
        if verbose:
            print("Input distribution is a numberic numpy array or tf.Tensor")
        if len(dist_input.shape) != 2:
            raise ValueError("Input must be a 2-dimensional numpy array or a tfp.distributions.Distribution object")
        else:
            dist_symb = None
            dist_num = tf.convert_to_tensor(dist_input)
            nsamples, ndims = dist_num.shape
            dtype = dist_num.dtype
    elif isinstance(dist_input, tfp.distributions.Distribution):
        if verbose:
            print("Input distribution is a tfp.distributions.Distribution object")
        dist_symb = dist_input
        dist_num = tf.convert_to_tensor([])
        nsamples, ndims = None, dist_symb.sample(2).numpy().shape[1]
        dtype = dist_symb.dtype
    else:
        raise ValueError("Input must be either a numpy array or a tfp.distributions.Distribution object")
    return dist_symb, dist_num, ndims, nsamples, dtype


def __parse_input_dist_tf(dist_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                        verbose: bool = False
                       ) -> Tuple[tf.Tensor, Optional[tfp.distributions.Distribution], tf.Tensor, int, Optional[int]]:
    
    def is_ndarray_or_tensor():
        return tf.reduce_any([isinstance(dist_input, np.ndarray), tf.is_tensor(dist_input)])
    
    def is_distribution():
        return tf.reduce_all([
            tf.logical_not(is_ndarray_or_tensor()),
            tf.reduce_any([isinstance(dist_input, tfp.distributions.Distribution)])
        ])

    def handle_distribution():
        conditional_tf_print(string = "Input distribution is a tfp.distributions.Distribution object", verbose = verbose)
        dist_symb: tf.distributions.Distribution = dist_input
        dist_num = tf.convert_to_tensor([[]],dtype=dist_symb.dtype)
        nsamples, ndims = tf.constant(0), tf.shape(dist_symb.sample(2))[1]
        return tf.constant(True), dist_symb, dist_num, ndims, nsamples

    def handle_ndarray_or_tensor():
        conditional_tf_print(string = "Input distribution is a numeric numpy array or tf.Tensor", verbose = verbose)
        if tf.rank(dist_input) != 2:
            tf.debugging.assert_equal(tf.rank(dist_input), 2, "Input must be a 2-dimensional numpy array or a tfp.distributions.Distribution object")
        dist_symb = None
        dist_num = tf.convert_to_tensor(dist_input)
        nsamples, ndims = tf.unstack(tf.shape(dist_num))
        return tf.constant(False), dist_symb, dist_num, ndims, nsamples

    def handle_else():
        tf.debugging.assert_equal(
            tf.reduce_any([is_distribution(), is_ndarray_or_tensor()]),
            True,
            "Input must be either a numpy array or a tfp.distributions.Distribution object"
        )

    conditional_tf_print(string = "Parsing input distribution...", verbose = verbose)

    return tf.case([
        (is_distribution(), handle_distribution),
        (is_ndarray_or_tensor(), handle_ndarray_or_tensor)
    ], default=handle_else, exclusive=True)


def KS_test_1_large(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    niter: int = 10,
                    batch_size: int = 100000,
                    dtype_input: Optional[tf.DType] = None,
                    use_tf: bool = False,
                    verbose: bool = False,
                    progress_bar: bool = False
                   ) -> Tuple[List[float], List[float]]:
    """
    The Kolmogorov-Smirnov test is a non-parametric test that compares two distributions and returns a test statistic and a p-value (the test statistic distribution is known a-priori) 
    that indicates whether the two distributions are the same or not.
    The test is performed for each dimension of the two distributions and an average is returned for both the test statistic and the p-value. This averages are used as test statistics whose
    distribution is estimated by repeating the test niter times. The niter list of test statistics (mean test_statistic and mean p-value) are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The ks_test_statistic_mean and ks_pvalue_mean are then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions KS_test_1 and KS_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, KS_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while KS_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but KS_test_1 performs independent 
    tests with less points, while KS_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        metric_list, pvalue_list: tuple of two lists of float, list of ks_test_statistic_mean and ks_pvalue_mean for each iteration.
    """    
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    metric_list: np.ndarray = np.zeros(niter)
    pvalue_list: np.ndarray = np.zeros(niter)
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if dtype_input is not None:
        dtype = dtype_input
    else:
        dtype = __get_best_dtype(dtype_1, dtype_2)
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if dist_1_symb is None:
            dist_1_num = tf.cast(dist_1_num[:nsamples,:], dtype = dtype)
        if dist_2_symb is None:
            dist_2_num = tf.cast(dist_2_num[:nsamples,:], dtype = dtype)
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar = tqdm(total = niter, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting KS tests calculation...")
        start: float = timer()
    
    if dist_1_symb is None and dist_2_symb is None:
        for k in range(niter):
            dist_1_k: tf.Tensor = dist_1_num[k*batch_size:(k+1)*batch_size,:]
            dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            [metric_list[k], pvalue_list[k]] = np.mean([ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
            if pbar:
                pbar.update(1)
    elif dist_1_symb is None and dist_2_symb is not None:
        for k in range(niter):
            dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
            dist_2_k = tf.cast(dist_2_symb.sample(batch_size), dtype = dtype)
            [metric_list[k], pvalue_list[k]] = np.mean([ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
            if pbar:
                pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is None:
        for k in range(niter):
            dist_1_k = tf.cast(dist_1_symb.sample(batch_size), dtype = dtype)
            dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            [metric_list[k], pvalue_list[k]] = np.mean([ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
            if pbar:
                pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is not None:
        for k in range(niter):
            dist_1_k = tf.cast(dist_1_symb.sample(batch_size), dtype = dtype)
            dist_2_k = tf.cast(dist_2_symb.sample(batch_size), dtype = dtype)
            [metric_list[k], pvalue_list[k]] = np.mean([ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
            if pbar:
                pbar.update(1)
                
    if verbose:
        end: float = timer()
        print("KS tests calculation completed in {:.2f} seconds".format(end-start))
                
    return metric_list.tolist(), pvalue_list.tolist()


def KS_test_1_small(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    niter: int = 10,
                    batch_size: int = 100000,
                    dtype_input: Optional[tf.DType] = None,
                    use_tf: bool = False,
                    verbose: bool = False,
                    progress_bar: bool = False
                   ) -> Tuple[List[float], List[float]]:
    """
    The Kolmogorov-Smirnov test is a non-parametric test that compares two distributions and returns a test statistic and a p-value (the test statistic distribution is known a-priori) 
    that indicates whether the two distributions are the same or not.
    The test is performed for each dimension of the two distributions and an average is returned for both the test statistic and the p-value. This averages are used as test statistics whose
    distribution is estimated by repeating the test niter times. The niter list of test statistics (mean test_statistic and mean p-value) are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The ks_test_statistic_mean and ks_pvalue_mean are then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions KS_test_1 and KS_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, KS_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while KS_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but KS_test_1 performs independent 
    tests with less points, while KS_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        metric_list, pvalue_list: tuple of two lists of float, list of ks_test_statistic_mean and ks_pvalue_mean for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    metric_list: np.ndarray = np.zeros(niter)
    pvalue_list: np.ndarray = np.zeros(niter)
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if dtype_input is not None:
        dtype = dtype_input
    else:
        dtype = __get_best_dtype(dtype_1, dtype_2)
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if dist_1_symb is None:
        dist_1_num = tf.cast(dist_1_num[:nsamples,:], dtype = dtype)
    else:
        dist_1_num = tf.cast(dist_1_symb.sample(batch_size*niter), dtype = dtype)
    if dist_2_symb is None:
        dist_2_num = tf.cast(dist_2_num[:nsamples,:], dtype = dtype)
    else:
        dist_2_num = tf.cast(dist_2_symb.sample(batch_size*niter), dtype = dtype)
    if nsamples is not None:
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if use_tf:
        ks_func = ks_2samp_tf
    else:
        ks_func = ks_2samp
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total=niter, desc="Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting KS tests calculation...")
        start: float = timer()
    
    for k in range(niter):
        dist_1_k: tf.Tensor = dist_1_num[k*batch_size:(k+1)*batch_size,:]
        dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
        [metric_list[k], pvalue_list[k]] = np.mean([ks_func(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
        if pbar:
            pbar.update(1)
            
    if verbose:
        end: float = timer()
        print("KS tests calculation completed in {:.2f} seconds".format(end-start))
                
    return metric_list.tolist(), pvalue_list.tolist()


def KS_test_1_small_tf(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                       dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                       niter: int = 10,
                       batch_size: int = 100000,
                       dtype_input: tf.DType = tf.float32,
                       use_tf: bool = False,
                       verbose: bool = False,
                       progress_bar: bool = False
                      ) -> tf.Tensor:
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims: int
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    pbar: Optional[tqdm]
    start: float
    end: float
    metric_list: np.ndarray = np.zeros(niter)
    pvalue_list: np.ndarray = np.zeros(niter)
    
    is_symb_1, dist_1_symb, dist_1_num, ndims_1, nsamples_1 = __parse_input_dist_tf(dist_1_input, verbose = verbose)
    is_symb_2, dist_2_symb, dist_2_num, ndims_2, nsamples_2 = __parse_input_dist_tf(dist_2_input, verbose = verbose)
    
    dtype_1 = dist_1_num.dtype
    dtype_2 = dist_2_num.dtype
    
    # Utility functions
    def set_ndims(value: int) -> None:
        nonlocal ndims
        ndims = value
        
    def raise_ndims_error() -> None:
        tf.debugging.assert_equal(ndims_1, ndims_2, message="dist_1 and dist_2 must have the same number of dimensions")
    
    def set_nsamples(value: int) -> None:
        nonlocal nsamples
        nsamples = value
        
    def set_dist_num_from_symb(dist: tfp.distributions.Distribution) -> tf.Tensor:
        nonlocal dtype
        dist_num = tf.cast(dist.sample(batch_size*niter), dtype=dtype)
        return dist_num
    
    def raise_batch_size_error() -> None:
        tf.debugging.assert_greater(nsamples, niter, message="nsamples must be greater than niter when at least one distribution is numerical")
    
    def set_batch_size(input: int) -> None:
        nonlocal batch_size
        batch_size = input
        tf.cond(tf.equal(batch_size, 0), true_fn = lambda: raise_batch_size_error(), false_fn = lambda: None)
        
    def start_calculation() -> None:
        nonlocal start, verbose
        conditional_tf_print(string = "Starting KS tests calculation...", verbose = verbose)
        start = timer()
        
    def init_progress_bar(niter: int) -> None:
        nonlocal pbar
        pbar = tqdm(total=niter, desc="Iterations")
        
    def update_progress_bar() -> None:
        nonlocal pbar
        pbar.update(1)
        
    def end_calculation() -> None:
        nonlocal end, verbose
        end = timer()
        conditional_tf_print(string = "KS tests calculation completed in "+str(end-start)+" seconds", verbose = verbose)
        
    # Check and set ndims
    tf.cond(tf.equal(ndims_1, ndims_2), true_fn = lambda: set_ndims(ndims_1), false_fn = lambda: raise_ndims_error())
    
    # Check and set dtype
    dtype = __get_best_dtype(dtype_input,__get_best_dtype(dtype_1, dtype_2))
    
    # Check and set nsamples
    tf.cond(tf.not_equal(nsamples_1, tf.constant(0)),
            true_fn = lambda: tf.cond(tf.not_equal(nsamples_2, tf.constant(0)),
                                       true_fn = lambda: set_nsamples(min(nsamples_1, nsamples_2)),
                                       false_fn = lambda: set_nsamples(nsamples_1)),
            false_fn = lambda: tf.cond(tf.not_equal(nsamples_2, tf.constant(0)),
                                       true_fn = lambda: set_nsamples(nsamples_2),
                                       false_fn = lambda: set_nsamples(tf.constant(0))))

    # Check and set distributions
    dist_1_num = tf.cond(tf.logical_not(is_symb_1),
                         true_fn = lambda: tf.cast(dist_1_num[:nsamples, :], dtype=dtype),
                         false_fn = lambda: set_dist_num_from_symb(dist_1_symb))
    dist_2_num = tf.cond(tf.logical_not(is_symb_2),
                         true_fn = lambda: tf.cast(dist_2_num[:nsamples, :], dtype=dtype),
                         false_fn = lambda: set_dist_num_from_symb(dist_2_symb))

    tf.cond(tf.less(nsamples, tf.constant(batch_size * niter)),
            true_fn = lambda: set_batch_size(nsamples // niter), # type: ignore
            false_fn = lambda: None)
               
    tf.cond(tf.equal(verbose, True), true_fn = lambda: start_calculation(), false_fn = lambda: None)
    tf.cond(tf.equal(progress_bar, True), true_fn = lambda: init_progress_bar(niter), false_fn = lambda: None)
        
    res = tf.TensorArray(dtype, size=niter)
        
    def body(k, res):
        dist_1_k = dist_1_num[k * batch_size:(k + 1) * batch_size, :]
        dist_2_k = dist_2_num[k * batch_size:(k + 1) * batch_size, :]
 
        ks_tests = tf.TensorArray(dtype, size=ndims)
        ks_pvalues = tf.TensorArray(dtype, size=ndims)
 
        def loop_body(dim, ks_tests, ks_pvalues):
            ks_result, ks_pvalue = tf.numpy_function(ks_2samp_wrapper, 
                                                     [dist_1_k[:, dim], dist_2_k[:, dim]], 
                                                     [dtype, dtype])
            ks_result = tf.cast(ks_result, dtype)
            ks_pvalue = tf.cast(ks_pvalue, dtype)
            ks_tests = ks_tests.write(dim, ks_result)
            ks_pvalues = ks_pvalues.write(dim, ks_pvalue)
            return dim + 1, ks_tests, ks_pvalues
 
        _, ks_tests, ks_pvalues = tf.while_loop(lambda dim, ks_tests, ks_pvalues: dim < ndims,
                                                loop_body,
                                                [0, ks_tests, ks_pvalues])
 
        mean_ks_tests_batch = tf.cast(tf.reduce_mean(ks_tests.stack()), dtype)
        mean_ks_pvalues_batch = tf.cast(tf.reduce_mean(ks_pvalues.stack()), dtype)
        res = res.write(k, tf.stack([mean_ks_tests_batch, mean_ks_pvalues_batch], axis=0))
 
        tf.cond(tf.equal(progress_bar, True), true_fn = lambda: update_progress_bar(), false_fn = lambda: None)
 
        return k + 1, res
 
    _, result = tf.while_loop(lambda k, res: k < niter, body, [0, res])
    
    tf.cond(tf.equal(progress_bar, True), true_fn = lambda: pbar.close(), false_fn = lambda: None)
    tf.cond(tf.equal(verbose, True), true_fn = lambda: end_calculation(), false_fn = lambda: None)
    
    return result.stack()


def KS_test_1(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
              dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
              niter: int = 10,
              batch_size: int = 100000,
              dtype_input: Optional[tf.DType] = None,
              use_tf: bool = False,
              verbose: bool = False,
              progress_bar: bool = False
             ) -> Tuple[List[float], List[float]]:
    _, _, ndims, _, _ = __parse_input_dist(dist_1_input, verbose = verbose)
    nn: int = batch_size*niter*ndims
    if nn < 1e7:
        try:
            if verbose:
                print("Total number of tests ",nn," < 1e7. Using the fastest KS_test_1_small.")
            return KS_test_1_small(dist_1_input, dist_2_input, niter, batch_size, dtype_input, use_tf, verbose, progress_bar)
        except Exception as e:
            print("Caught an exception in KS_test_1_small. Details:")
            traceback.print_exc()
            if verbose:
                print("Failed to use KS_test_1_small. Using KS_test_1_large.")
            return KS_test_1_large(dist_1_input, dist_2_input, niter, batch_size, dtype_input, use_tf, verbose, progress_bar)
    else:
        if verbose:
            print("Total number of tests ",nn," >= 1e7. Using KS_test_1_large.")
        return KS_test_1_large(dist_1_input, dist_2_input, niter, batch_size, use_tf, verbose, progress_bar)
    

def KS_test_2_large(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    niter: int = 10,
                    batch_size: int = 100000,
                    dtype_input: Optional[tf.DType] = None,
                    use_tf: bool = False,
                    verbose: bool = False,
                    progress_bar: bool = False
                   ) -> Tuple[List[float], List[float]]:
    """
    The Kolmogorov-Smirnov test is a non-parametric test that compares two distributions and returns a test statistic and a p-value (the test statistic distribution is known a-priori) 
    that indicates whether the two distributions are the same or not.
    The test is performed for each dimension of the two distributions and an average is returned for both the test statistic and the p-value. This averages are used as test statistics whose
    distribution is estimated by repeating the test niter times. The niter list of test statistics (mean test_statistic and mean p-value) are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The ks_test_statistic_mean and ks_pvalue_mean are then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions KS_test_1 and KS_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, KS_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while KS_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but KS_test_1 performs independent 
    tests with less points, while KS_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        metric_list, pvalue_list: tuple of two lists of float, list of ks_test_statistic_mean and ks_pvalue_mean for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    metric_list: np.ndarray = np.zeros(niter)
    pvalue_list: np.ndarray = np.zeros(niter)
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if dtype_input is not None:
        dtype = dtype_input
    else:
        dtype = __get_best_dtype(dtype_1, dtype_2)
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if dist_1_symb is None:
            dist_1_num = tf.cast(dist_1_num[:nsamples,:], dtype = dtype)
        if dist_2_symb is None:
            dist_2_num = tf.cast(dist_2_num[:nsamples,:], dtype = dtype)
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if use_tf:
        ks_func = ks_2samp_tf
    else:
        ks_func = ks_2samp
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total = niter**2, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting KS tests calculation...")
        start: float = timer()
    
    l: int = 0
    if dist_1_symb is None and dist_2_symb is None:
        for j in range(niter):
            dist_1_j: tf.Tensor = dist_1_num[j*batch_size:(j+1)*batch_size,:]
            for k in range(niter):
                dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
                [metric_list[l], pvalue_list[l]] = np.mean([ks_func(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is None and dist_2_symb is not None:
        for j in range(niter):
            dist_1_j = dist_1_num[j*batch_size:(j+1)*batch_size,:]
            for k in range(niter):
                dist_2_k = tf.cast(dist_2_symb.sample(batch_size), dtype = dtype)
                [metric_list[l], pvalue_list[l]] = np.mean([ks_func(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is None:
        for j in range(niter):
            dist_1_j = tf.cast(dist_1_symb.sample(batch_size), dtype = dtype)
            for k in range(niter):
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]    
                [metric_list[l], pvalue_list[l]] = np.mean([ks_func(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is not None:
        for j in range(niter):
            dist_1_j = tf.cast(dist_1_symb.sample(batch_size), dtype = dtype)
            for k in range(niter):
                dist_2_k = tf.cast(dist_2_symb.sample(batch_size), dtype = dtype)
                [metric_list[l], pvalue_list[l]] = np.mean([ks_func(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
                l += 1
                if pbar:
                    pbar.update(1)
                    
    if verbose:
        end: float = timer()
        print("KS tests calculation completed in {:.2f} seconds".format(end-start))

    return metric_list.tolist(), pvalue_list.tolist()


def KS_test_2_small(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    niter: int = 10,
                    batch_size: int = 100000,
                    dtype_input: Optional[tf.DType] = None,
                    use_tf: bool = False,
                    verbose: bool = False,
                    progress_bar: bool = False
                   ) -> Tuple[List[float], List[float]]:
    """
    The Kolmogorov-Smirnov test is a non-parametric test that compares two distributions and returns a test statistic and a p-value (the test statistic distribution is known a-priori) 
    that indicates whether the two distributions are the same or not.
    The test is performed for each dimension of the two distributions and an average is returned for both the test statistic and the p-value. This averages are used as test statistics whose
    distribution is estimated by repeating the test niter times. The niter list of test statistics (mean test_statistic and mean p-value) are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The ks_test_statistic_mean and ks_pvalue_mean are then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions KS_test_1 and KS_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, KS_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while KS_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but KS_test_1 performs independent 
    tests with less points, while KS_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        metric_list, pvalue_list: tuple of two lists of float, list of ks_test_statistic_mean and ks_pvalue_mean for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    metric_list: np.ndarray = np.zeros(niter)
    pvalue_list: np.ndarray = np.zeros(niter)
        
    niter = int(np.ceil(np.sqrt(niter)))
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if dtype_input is not None:
        dtype = dtype_input
    else:
        dtype = __get_best_dtype(dtype_1, dtype_2)
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if dist_1_symb is None:
        dist_1_num = tf.cast(dist_1_num[:nsamples,:], dtype = dtype)
    else:
        dist_1_num = tf.cast(dist_1_symb.sample(batch_size*niter), dtype = dtype)
    if dist_2_symb is None:
        dist_2_num = tf.cast(dist_2_num[:nsamples,:], dtype = dtype)
    else:
        dist_2_num = tf.cast(dist_2_symb.sample(batch_size*niter), dtype = dtype)
    if nsamples is not None:
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if use_tf:
        ks_func = ks_2samp_tf
    else:
        ks_func = ks_2samp
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total = niter**2, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting KS tests calculation...")
        start: float = timer()
    
    l: int = 0
    for j in range(niter):
        dist_1_j: tf.Tensor = dist_1_num[j*batch_size:(j+1)*batch_size,:]
        for k in range(niter):
            dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            [metric_list[l], pvalue_list[l]] = np.mean([ks_func(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
            l += 1
            if pbar:
                pbar.update(1)
                
    if verbose:
        end: float = timer()
        print("KS tests calculation completed in {:.2f} seconds".format(end-start))
            
    return metric_list.tolist(), pvalue_list.tolist()


def KS_test_2(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
              dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
              niter: int = 10,
              batch_size: int = 100000,
              dtype_input: Optional[tf.DType] = None,
              use_tf: bool = False,
              verbose: bool = False,
              progress_bar: bool = False
             ) -> Tuple[List[float], List[float]]:
    _, _, ndims, _, _ = __parse_input_dist(dist_1_input, verbose = verbose)
    nn: int = batch_size*niter*ndims
    if nn < 1e7:
        try:
            if verbose:
                print("Total number of tests ",nn," < 1e7. Using the fastest KS_test_2_small.")
            return KS_test_2_small(dist_1_input, dist_2_input, niter, batch_size, dtype_input, use_tf, verbose, progress_bar)
        except Exception as e:
            print("Caught an exception in KS_test_2_small. Details:")
            traceback.print_exc()
            if verbose:
                print("Failed to use KS_test_2_small. Using KS_test_2_large.")
            return KS_test_2_large(dist_1_input, dist_2_input, niter, batch_size, dtype_input, use_tf, verbose, progress_bar)
    else:
        if verbose:
            print("Total number of tests ",nn," >= 1e7. Using KS_test_2_large.")
        return KS_test_2_large(dist_1_input, dist_2_input, niter, batch_size, dtype_input, use_tf, verbose, progress_bar)
    

def AD_test_1_large(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    niter: int = 10,
                    batch_size: int = 100000,
                    verbose: bool = False,
                    progress_bar: bool = False
                   ) -> Tuple[List[float], List[float]]:
    """
    The Anderson-Darling test is a non-parametric test that compares two distributions and returns a test statistic and a p-value (the test statistic distribution is known a-priori) 
    that indicates whether the two distributions are the same or not.
    The test is performed for each dimension of the two distributions and an average is returned for both the test statistic and the p-value. This averages are used as test statistics whose
    distribution is estimated by repeating the test niter times. The niter list of test statistics (mean test_statistic and mean p-value) are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The ad_test_statistic_mean and ks_pvalue_mean are then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions AD_test_1 and AD_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, AD_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while AD_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but AD_test_1 performs independent 
    tests with less points, while AD_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        metric_list, pvalue_list: tuple of two lists of float, list of ad_test_statistic_mean and ad_pvalue_mean for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    metric_list: np.ndarray = np.zeros(niter)
    pvalue_list: np.ndarray = np.zeros(niter)
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if dist_1_symb is None:
            dist_1_num = dist_1_num[:nsamples,:]
        if dist_2_symb is None:
            dist_2_num = dist_2_num[:nsamples,:]
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar = tqdm(total = niter, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting AD tests calculation...")
        start: float = timer()
    
    if dist_1_symb is None and dist_2_symb is None:
        for k in range(niter):
            dist_1_k: tf.Tensor = dist_1_num[k*batch_size:(k+1)*batch_size,:]
            dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            test = [anderson_ksamp([dist_1_k[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
            [metric_list[k], pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
            if pbar:
                pbar.update(1)
    elif dist_1_symb is None and dist_2_symb is not None:
        for k in range(niter):
            dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
            dist_2_k = dist_2_symb.sample(batch_size)
            test = [anderson_ksamp([dist_1_k[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
            [metric_list[k], pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
            if pbar:
                pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is None:
        for k in range(niter):
            dist_1_k = dist_1_symb.sample(batch_size)
            dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            test = [anderson_ksamp([dist_1_k[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
            [metric_list[k], pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
            if pbar:
                pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is not None:
        for k in range(niter):
            dist_1_k = dist_1_symb.sample(batch_size)
            dist_2_k = dist_2_symb.sample(batch_size)
            test = [anderson_ksamp([dist_1_k[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
            [metric_list[k], pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
            if pbar:
                pbar.update(1)
    
    if verbose:
        end: float = timer()
        print("AD tests calculation completed in {:.2f} seconds".format(end-start))

    return metric_list.tolist(), pvalue_list.tolist()


def AD_test_1_small(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    niter: int = 10,
                    batch_size: int = 100000,
                    verbose: bool = False,
                    progress_bar: bool = False
                   ) -> Tuple[List[float], List[float]]:
    """
    The Anderson-Darling test is a non-parametric test that compares two distributions and returns a test statistic and a p-value (the test statistic distribution is known a-priori) 
    that indicates whether the two distributions are the same or not.
    The test is performed for each dimension of the two distributions and an average is returned for both the test statistic and the p-value. This averages are used as test statistics whose
    distribution is estimated by repeating the test niter times. The niter list of test statistics (mean test_statistic and mean p-value) are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The ad_test_statistic_mean and ks_pvalue_mean are then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions AD_test_1 and AD_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, AD_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while AD_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but AD_test_1 performs independent 
    tests with less points, while AD_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        metric_list, pvalue_list: tuple of two lists of float, list of ad_test_statistic_mean and ad_pvalue_mean for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    metric_list: np.ndarray = np.zeros(niter)
    pvalue_list: np.ndarray = np.zeros(niter)
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if dist_1_symb is None:
        dist_1_num = dist_1_num[:nsamples,:]
    else:
        dist_1_num = dist_1_symb.sample(batch_size*niter)
    if dist_2_symb is None:
        dist_2_num = dist_2_num[:nsamples,:]
    else:
        dist_2_num = dist_2_symb.sample(batch_size*niter)
    if nsamples is not None:
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar = tqdm(total = niter, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting AD tests calculation...")
        start: float = timer()

    for k in range(niter):
        dist_1_k: tf.Tensor = dist_1_num[k*batch_size:(k+1)*batch_size,:]
        dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
        test = [anderson_ksamp([dist_1_k[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
        [metric_list[k],pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
        if pbar:
            pbar.update(1)
            
    if verbose:
        end: float = timer()
        print("AD tests calculation completed in {:.2f} seconds".format(end-start))
        
    return metric_list.tolist(), pvalue_list.tolist()


def AD_test_1(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
              dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
              niter: int = 10,
              batch_size: int = 100000,
              verbose: bool = False,
              progress_bar: bool = False
             ) -> Tuple[List[float], List[float]]:
    _, _, ndims, _, _ = __parse_input_dist(dist_1_input, verbose = verbose)
    nn: int = batch_size*niter*ndims
    if nn < 1e7:
        try:
            return AD_test_1_small(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
        except:
            return AD_test_1_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
    else:
        return AD_test_1_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
    

def AD_test_2_large(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    niter: int = 10,
                    batch_size: int = 100000,
                    verbose: bool = False,
                    progress_bar: bool = False
                   ) -> Tuple[List[float], List[float]]:
    """
    The Anderson-Darling test is a non-parametric test that compares two distributions and returns a test statistic and a p-value (the test statistic distribution is known a-priori) 
    that indicates whether the two distributions are the same or not.
    The test is performed for each dimension of the two distributions and an average is returned for both the test statistic and the p-value. This averages are used as test statistics whose
    distribution is estimated by repeating the test niter times. The niter list of test statistics (mean test_statistic and mean p-value) are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The ad_test_statistic_mean and ks_pvalue_mean are then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions AD_test_1 and AD_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, AD_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while AD_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but AD_test_1 performs independent 
    tests with less points, while AD_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        metric_list, pvalue_list: tuple of two lists of float, list of ad_test_statistic_mean and ad_pvalue_mean for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    metric_list: np.ndarray = np.zeros(niter)
    pvalue_list: np.ndarray = np.zeros(niter)
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if dist_1_symb is None:
            dist_1_num = dist_1_num[:nsamples,:]
        if dist_2_symb is None:
            dist_2_num = dist_2_num[:nsamples,:]
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total = niter**2, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting AD tests calculation...")
        start: float = timer()
    
    l: int = 0
    if dist_1_symb is None and dist_2_symb is None:
        for j in range(niter):
            dist_1_j: tf.Tensor = dist_1_num[j*batch_size:(j+1)*batch_size,:]
            for k in range(niter):
                dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
                test = [anderson_ksamp([dist_1_j[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
                [metric_list[l], pvalue_list[l]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is None and dist_2_symb is not None:
        for j in range(niter):
            dist_1_j = dist_1_num[j*batch_size:(j+1)*batch_size,:]
            for k in range(niter):
                dist_2_k = dist_2_symb.sample(batch_size)
                test = [anderson_ksamp([dist_1_j[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
                [metric_list[l], pvalue_list[l]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is None:
        for j in range(niter):
            dist_1_j = dist_1_symb.sample(batch_size)
            for k in range(niter):
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]  
                test = [anderson_ksamp([dist_1_j[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
                [metric_list[l], pvalue_list[l]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is not None:
        for j in range(niter):
            dist_1_j = dist_1_symb.sample(batch_size)
            for k in range(niter):
                dist_2_k = dist_2_symb.sample(batch_size)
                test = [anderson_ksamp([dist_1_j[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
                [metric_list[l], pvalue_list[l]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
                l += 1
                if pbar:
                    pbar.update(1)
                    
    if verbose:
        end: float = timer()
        print("AD tests calculation completed in {:.2f} seconds".format(end-start))
                
    return metric_list.tolist(), pvalue_list.tolist()


def AD_test_2_small(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                    niter: int = 10,
                    batch_size: int = 100000,
                    verbose: bool = False,
                    progress_bar: bool = False
                   ) -> Tuple[List[float], List[float]]:
    """
    The Anderson-Darling test is a non-parametric test that compares two distributions and returns a test statistic and a p-value (the test statistic distribution is known a-priori) 
    that indicates whether the two distributions are the same or not.
    The test is performed for each dimension of the two distributions and an average is returned for both the test statistic and the p-value. This averages are used as test statistics whose
    distribution is estimated by repeating the test niter times. The niter list of test statistics (mean test_statistic and mean p-value) are returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The ad_test_statistic_mean and ks_pvalue_mean are then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions AD_test_1 and AD_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, AD_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while AD_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but AD_test_1 performs independent 
    tests with less points, while AD_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        metric_list, pvalue_list: tuple of two lists of float, list of ad_test_statistic_mean and ad_pvalue_mean for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    metric_list: np.ndarray = np.zeros(niter)
    pvalue_list: np.ndarray = np.zeros(niter)
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if dist_1_symb is None:
        dist_1_num = dist_1_num[:nsamples,:]
    else:
        dist_1_num = dist_1_symb.sample(batch_size*niter)
    if dist_2_symb is None:
        dist_2_num = dist_2_num[:nsamples,:]
    else:
        dist_2_num = dist_2_symb.sample(batch_size*niter)
    if nsamples is not None:
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total = niter**2, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting AD tests calculation...")
        start: float = timer()
    
    l: int = 0  
    for j in range(niter):
        dist_1_j: tf.Tensor = dist_1_num[j*batch_size:(j+1)*batch_size,:]
        for k in range(niter):
            dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            test = [anderson_ksamp([dist_1_j[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
            [metric_list[l], pvalue_list[l]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
            if pbar:
                pbar.update(1)
                
    if verbose:
        end: float = timer()
        print("AD tests calculation completed in {:.2f} seconds".format(end-start))
            
    return metric_list.tolist(), pvalue_list.tolist()


def AD_test_2(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
              dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
              niter: int = 10,
              batch_size: int = 100000,
              verbose: bool = False,
              progress_bar: bool = False
             ) -> Tuple[List[float], List[float]]:
    _, _, ndims, _, _ = __parse_input_dist(dist_1_input, verbose = verbose)
    nn: int = batch_size*niter*ndims
    if nn < 1e7:
        try:
            return AD_test_2_small(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
        except:
            return AD_test_2_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
    else:
        return AD_test_2_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
    

def FN_1_large(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               niter: int = 10,
               batch_size: int = 100000,
               verbose: bool = False,
               progress_bar: bool = False
              ) -> List[float]:
    """
    The Frobenius-Norm of the difference between the correlation matrices of two distributions.
    The value of this metric is computed niter times and a list of the fn_values is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The fn_values are then computed over all pairs of batches dist_1_j, dist_2_j and the list is returned.
    
    NOTE: The functions FN_1 and FN_2 differ in the way batches are computed and compared. For instanve, for 10 iterations with 100000 samples, 
    FN_1 generates 10 pairs of independent batches of 10000 samples each and performs the tests pairwise (10 tests with 10000 points), while FN_2 generates 3 pairs of independent batches of 100000/3
    samples each and performs the tests for all pairs (9 tests with 100000/3 points). In the case of FN_1 tests are done with less points, but are independent, while in the case of 
    FN_2 tests are done with more points, but are not independent.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        fn_values_list: lists of floats, list of fn_values for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    values_list: np.ndarray = np.zeros(niter)
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if dist_1_symb is None:
            dist_1_num = dist_1_num[:nsamples,:]
        if dist_2_symb is None:
            dist_2_num = dist_2_num[:nsamples,:]
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar = tqdm(total = niter, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting FN metric calculation...")
        start: float = timer()
    
    if dist_1_symb is None and dist_2_symb is None:
        for k in range(niter):
            dist_1_k: tf.Tensor = dist_1_num[k*batch_size:(k+1)*batch_size,:]
            dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_k, sample_axis=0, event_axis=-1))
            dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
            values_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
            if pbar:
                pbar.update(1)
    elif dist_1_symb is None and dist_2_symb is not None:
        for k in range(niter):
            dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
            dist_2_k = dist_2_symb.sample(batch_size)
            dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_k, sample_axis=0, event_axis=-1))
            dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
            values_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
            if pbar:
                pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is None:
        for k in range(niter):
            dist_1_k = dist_1_symb.sample(batch_size)
            dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_k, sample_axis=0, event_axis=-1))
            dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
            values_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
            if pbar:
                pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is not None:
        for k in range(niter):
            dist_1_k = dist_1_symb.sample(batch_size)
            dist_2_k = dist_2_symb.sample(batch_size)
            dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_k, sample_axis=0, event_axis=-1))
            dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
            values_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
            if pbar:
                pbar.update(1)
                
    if verbose: 
        end: float = timer()
        print("FN metric calculation completed in {:.2f} seconds".format(end-start))

    return values_list.tolist()


def FN_1_small(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               niter: int = 10,
               batch_size: int = 100000,
               verbose: bool = False,
               progress_bar: bool = False
              ) -> List[float]:
    """
    The Frobenius-Norm of the difference between the correlation matrices of two distributions.
    The value of this metric is computed niter times and a list of the fn_values is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The fn_values are then computed over all pairs of batches dist_1_j, dist_2_j and the list is returned.
    
    NOTE: The functions FN_1 and FN_2 differ in the way batches are computed and compared. For instanve, for 10 iterations with 100000 samples, 
    FN_1 generates 10 pairs of independent batches of 10000 samples each and performs the tests pairwise (10 tests with 10000 points), while FN_2 generates 3 pairs of independent batches of 100000/3
    samples each and performs the tests for all pairs (9 tests with 100000/3 points). In the case of FN_1 tests are done with less points, but are independent, while in the case of 
    FN_2 tests are done with more points, but are not independent.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        fn_values_list: lists of floats, list of fn_values for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    values_list: np.ndarray = np.zeros(niter)
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if dist_1_symb is None:
        dist_1_num = dist_1_num[:nsamples,:]
    else:
        dist_1_num = dist_1_symb.sample(batch_size*niter)
    if dist_2_symb is None:
        dist_2_num = dist_2_num[:nsamples,:]
    else:
        dist_2_num = dist_2_symb.sample(batch_size*niter)
    if nsamples is not None:
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar = tqdm(total = niter, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting FN metric calculation...")
        start: float = timer()

    for k in range(niter):
        dist_1_k: tf.Tensor = dist_1_num[k*batch_size:(k+1)*batch_size,:]
        dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
        dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_k, sample_axis=0, event_axis=-1))
        dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
        values_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
        if pbar:
            pbar.update(1)
            
    if verbose: 
        end: float = timer()
        print("FN metric calculation completed in {:.2f} seconds".format(end-start))
        
    return values_list.tolist()


def FN_1(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
         dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
         niter: int = 10,
         batch_size: int = 100000,
         verbose: bool = False,
         progress_bar: bool = False
        ) -> List[float]:
    _, _, ndims, _, _ = __parse_input_dist(dist_1_input, verbose = verbose)
    nn: int = batch_size*niter*ndims
    if nn < 1e7:
        try:
            return FN_1_small(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
        except:
            return FN_1_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
    else:
        return FN_1_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
    

def FN_2_large(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               niter: int = 10,
               batch_size: int = 100000,
               verbose: bool = False,
               progress_bar: bool = False
              ) -> List[float]:
    """
    The Frobenius-Norm of the difference between the correlation matrices of two distributions.
    The value of this metric is computed niter times and a list of the fn_values is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The fn_values are then computed over all pairs of batches dist_1_j, dist_2_j and the list is returned.
    
    NOTE: The functions FN_1 and FN_2 differ in the way batches are computed and compared. For instanve, for 10 iterations with 100000 samples, 
    FN_1 generates 10 pairs of independent batches of 10000 samples each and performs the tests pairwise (10 tests with 10000 points), while FN_2 generates 3 pairs of independent batches of 100000/3
    samples each and performs the tests for all pairs (9 tests with 100000/3 points). In the case of FN_1 tests are done with less points, but are independent, while in the case of 
    FN_2 tests are done with more points, but are not independent.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        fn_values_list: lists of floats, list of fn_values for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    values_list: np.ndarray = np.zeros(niter)
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if dist_1_symb is None:
            dist_1_num = dist_1_num[:nsamples,:]
        if dist_2_symb is None:
            dist_2_num = dist_2_num[:nsamples,:]
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total = niter**2, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting FN metric calculation...")
        start: float = timer()
    
    l: int = 0
    if dist_1_symb is None and dist_2_symb is None:
        for j in range(niter):
            dist_1_j: tf.Tensor = dist_1_num[j*batch_size:(j+1)*batch_size,:]
            for k in range(niter):
                dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
                dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_j, sample_axis=0, event_axis=-1))
                dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
                values_list[l] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is None and dist_2_symb is not None:
        for j in range(niter):
            dist_1_j = dist_1_num[j*batch_size:(j+1)*batch_size,:]
            for k in range(niter):
                dist_2_k = dist_2_symb.sample(batch_size)
                dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_j, sample_axis=0, event_axis=-1))
                dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
                values_list[l] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is None:
        for j in range(niter):
            dist_1_j = dist_1_symb.sample(batch_size)
            for k in range(niter):
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]  
                dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_j, sample_axis=0, event_axis=-1))
                dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
                values_list[l] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is not None:
        for j in range(niter):
            dist_1_j = dist_1_symb.sample(batch_size)
            for k in range(niter):
                dist_2_k = dist_2_symb.sample(batch_size)
                dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_j, sample_axis=0, event_axis=-1))
                dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
                values_list[l] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
                l += 1
                if pbar:
                    pbar.update(1)
                    
    if verbose: 
        end: float = timer()
        print("FN metric calculation completed in {:.2f} seconds".format(end-start))

    return values_list.tolist()


def FN_2_small(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               niter: int = 10,
               batch_size: int = 100000,
               verbose: bool = False,
               progress_bar: bool = False
              ) -> List[float]:
    """
    The Frobenius-Norm of the difference between the correlation matrices of two distributions.
    The value of this metric is computed niter times and a list of the fn_values is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The fn_values are then computed over all pairs of batches dist_1_j, dist_2_j and the list is returned.
    
    NOTE: The functions FN_1 and FN_2 differ in the way batches are computed and compared. For instanve, for 10 iterations with 100000 samples, 
    FN_1 generates 10 pairs of independent batches of 10000 samples each and performs the tests pairwise (10 tests with 10000 points), while FN_2 generates 3 pairs of independent batches of 100000/3
    samples each and performs the tests for all pairs (9 tests with 100000/3 points). In the case of FN_1 tests are done with less points, but are independent, while in the case of 
    FN_2 tests are done with more points, but are not independent.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        fn_values_list: lists of floats, list of fn_values for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    values_list: np.ndarray = np.zeros(niter)
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if dist_1_symb is None:
        dist_1_num = dist_1_num[:nsamples,:]
    else:
        dist_1_num = dist_1_symb.sample(batch_size*niter)
    if dist_2_symb is None:
        dist_2_num = dist_2_num[:nsamples,:]
    else:
        dist_2_num = dist_2_symb.sample(batch_size*niter)
    if nsamples is not None:
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total = niter**2, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting FN metric calculation...")
        start: float = timer()
    
    l: int = 0  
    for j in range(niter):
        dist_1_j: tf.Tensor = dist_1_num[j*batch_size:(j+1)*batch_size,:]
        for k in range(niter):
            dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_j, sample_axis=0, event_axis=-1))
            dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
            values_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
            l += 1
            if pbar:
                pbar.update(1)
                
    if verbose: 
        end: float = timer()
        print("FN metric calculation completed in {:.2f} seconds".format(end-start))

    return values_list.tolist()


def FN_2(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
         dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
         niter: int = 10,
         batch_size: int = 100000,
         verbose: bool = False,
         progress_bar: bool = False  
        ) -> List[float]:
    _, _, ndims, _, _ = __parse_input_dist(dist_1_input, verbose = verbose)
    nn: int = batch_size*niter*ndims
    if nn < 1e7:
        try:
            return FN_2_small(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
        except:
            return FN_2_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
    else:
        return FN_2_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
    

def WD_1_large(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               niter: int = 10,
               batch_size: int = 100000,
               verbose: bool = False,
               progress_bar: bool = False
              ) -> List[float]:
    """
    The Wasserstein distance between two 1D distributions.
    The value of the distance is computed for each dimension of the two distributions and an average is returned. This average is used as a test statistic whose
    distribution is estimated by computing the quantity niter times. The niter list of test statistic is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The wd_mean_list is then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions WD_1 and WD_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, WD_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while WD_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but WD_1 performs independent 
    tests with less points, while WD_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        wd_mean_list: lists of float, list of wd_mean values for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    values_list: np.ndarray = np.zeros(niter)
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if dist_1_symb is None:
            dist_1_num = dist_1_num[:nsamples,:]
        if dist_2_symb is None:
            dist_2_num = dist_2_num[:nsamples,:]
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar = tqdm(total = niter, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting WD metric calculation...")
        start: float = timer()
    
    if dist_1_symb is None and dist_2_symb is None:
        for k in range(niter):
            dist_1_k: tf.Tensor = dist_1_num[k*batch_size:(k+1)*batch_size,:]
            dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]  
            values_list[k] = np.mean([wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
            if pbar:
                pbar.update(1)
    elif dist_1_symb is None and dist_2_symb is not None:
        for k in range(niter):
            dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
            dist_2_k = dist_2_symb.sample(batch_size)  
            values_list[k] = np.mean([wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
            if pbar:
                pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is None:
        for k in range(niter):
            dist_1_k = dist_1_symb.sample(batch_size)
            dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]  
            values_list[k] = np.mean([wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
            if pbar:
                pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is not None:
        for k in range(niter):
            dist_1_k = dist_1_symb.sample(batch_size)
            dist_2_k = dist_2_symb.sample(batch_size)  
            values_list[k] = np.mean([wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
            if pbar:
                pbar.update(1)
                
    if verbose:
        end: float = timer()
        print("WD metric calculation completed in {:.2f} seconds".format(end-start))

    return values_list.tolist()


def WD_1_small(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               niter: int = 10,
               batch_size: int = 100000,
               verbose: bool = False,
               progress_bar: bool = False
              ) -> List[float]:
    """
    The Wasserstein distance between two 1D distributions.
    The value of the distance is computed for each dimension of the two distributions and an average is returned. This average is used as a test statistic whose
    distribution is estimated by computing the quantity niter times. The niter list of test statistic is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The wd_mean_list is then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions WD_1 and WD_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, WD_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while WD_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but WD_1 performs independent 
    tests with less points, while WD_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        wd_mean_list: lists of float, list of wd_mean values for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    values_list: np.ndarray = np.zeros(niter)
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if dist_1_symb is None:
        dist_1_num = dist_1_num[:nsamples,:]
    else:
        dist_1_num = dist_1_symb.sample(batch_size*niter)
    if dist_2_symb is None:
        dist_2_num = dist_2_num[:nsamples,:]
    else:
        dist_2_num = dist_2_symb.sample(batch_size*niter)
    if nsamples is not None:
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar = tqdm(total = niter, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting WD metric calculation...")
        start: float = timer()
    
    for k in range(niter):
        dist_1_k: tf.Tensor = dist_1_num[k*batch_size:(k+1)*batch_size,:]
        dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
        values_list[k] = np.mean([wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
        if pbar:
            pbar.update(1)
            
    if verbose:
        end: float = timer()
        print("WD metric calculation completed in {:.2f} seconds".format(end-start))
        
    return values_list.tolist()


def WD_1(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
         dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
         niter: int = 10,
         batch_size: int = 100000,
         verbose: bool = False,
         progress_bar: bool = False
        ) -> List[float]:
    _, _, ndims, _, _ = __parse_input_dist(dist_1_input, verbose = verbose)
    nn: int = batch_size*niter*ndims
    if nn < 1e7:
        try:
            return WD_1_small(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
        except:
            return WD_1_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
    else:
        return WD_1_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)

        
def WD_2_large(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               niter: int = 10,
               batch_size: int = 100000,
               verbose: bool = False,
               progress_bar: bool = False
              ) -> List[float]:
    """
    The Wasserstein distance between two 1D distributions.
    The value of the distance is computed for each dimension of the two distributions and an average is returned. This average is used as a test statistic whose
    distribution is estimated by computing the quantity niter times. The niter list of test statistic is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The wd_mean_list is then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions WD_1 and WD_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, WD_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while WD_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but WD_1 performs independent 
    tests with less points, while WD_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        wd_mean_list: lists of float, list of wd_mean values for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    values_list: np.ndarray = np.zeros(niter)
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if dist_1_symb is None:
            dist_1_num = dist_1_num[:nsamples,:]
        if dist_2_symb is None:
            dist_2_num = dist_2_num[:nsamples,:]
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total = niter**2, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting WD metric calculation...")
        start: float = timer()
    
    l: int = 0
    if dist_1_symb is None and dist_2_symb is None:
        for j in range(niter):
            dist_1_j: tf.Tensor = dist_1_num[j*batch_size:(j+1)*batch_size,:]
            for k in range(niter):
                dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
                values_list[l] = np.mean([wasserstein_distance(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is None and dist_2_symb is not None:
        for j in range(niter):
            dist_1_j = dist_1_num[j*batch_size:(j+1)*batch_size,:]
            for k in range(niter):
                dist_2_k = dist_2_symb.sample(batch_size)
                values_list[l] = np.mean([wasserstein_distance(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is None:
        for j in range(niter):
            dist_1_j = dist_1_symb.sample(batch_size)
            for k in range(niter):
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]  
                values_list[l] = np.mean([wasserstein_distance(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is not None:
        for j in range(niter):
            dist_1_j = dist_1_symb.sample(batch_size)
            for k in range(niter):
                dist_2_k = dist_2_symb.sample(batch_size)
                values_list[l] = np.mean([wasserstein_distance(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
                l += 1
                if pbar:
                    pbar.update(1)
                    
    if verbose:
        end: float = timer()
        print("WD metric calculation completed in {:.2f} seconds".format(end-start))

    return values_list.tolist()


def WD_2_small(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
               niter: int = 10,
               batch_size: int = 100000,
               verbose: bool = False,
               progress_bar: bool = False
              ) -> List[float]:
    """
    The Wasserstein distance between two 1D distributions.
    The value of the distance is computed for each dimension of the two distributions and an average is returned. This average is used as a test statistic whose
    distribution is estimated by computing the quantity niter times. The niter list of test statistic is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The wd_mean_list is then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions WD_1 and WD_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, WD_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while WD_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but WD_1 performs independent 
    tests with less points, while WD_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        
    Returns:
        wd_mean_list: lists of float, list of wd_mean values for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    values_list: np.ndarray = np.zeros(niter)
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if dist_1_symb is None:
        dist_1_num = dist_1_num[:nsamples,:]
    else:
        dist_1_num = dist_1_symb.sample(batch_size*niter)
    if dist_2_symb is None:
        dist_2_num = dist_2_num[:nsamples,:]
    else:
        dist_2_num = dist_2_symb.sample(batch_size*niter)
    if nsamples is not None:
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total = niter**2, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting WD metric calculation...")
        start: float = timer()
    
    l: int = 0  
    for j in range(niter):
        dist_1_j: tf.Tensor = dist_1_num[j*batch_size:(j+1)*batch_size,:]
        for k in range(niter):
            dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            values_list[l] = np.mean([wasserstein_distance(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
            l += 1
            if pbar:
                pbar.update(1) # type: ignore
                
    if verbose:
        end: float = timer()
        print("WD metric calculation completed in {:.2f} seconds".format(end-start))

    return values_list.tolist()
        

def WD_2(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
         dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
         niter: int = 10,
         batch_size: int = 100000,
         verbose: bool = False,
         progress_bar: bool = False
        ) -> List[float]:
    _, _, ndims, _, _ = __parse_input_dist(dist_1_input, verbose = verbose)
    nn: int = batch_size*niter*ndims
    if nn < 1e7:
        try:
            return WD_2_small(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
        except:
            return WD_2_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
    else:
        return WD_2_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)


def SWD_1_large(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                niter: int = 10,
                batch_size: int = 100000,
                nslices: int = 100,
                seed: Optional[int] = None,
                verbose: bool = False,
                progress_bar: bool = False
               ) -> List[float]:
    """
    Compute the sliced Wasserstein distance between two multi-dimensional distribution as the average over slices of the value of the 1D Wasserstein distances 
    between sets of points obtained projecting data along nslices random directions.
    The SWD value is used as a test statistic whose distribution is estimated by computing the quantity niter times. The niter list of SWD test statistic is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The swd_list is then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions SWD_test_1 and SWD_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, SWD_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while SWD_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but SWD_test_1 performs independent 
    tests with less points, while SWD_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        nslices: int, number of random directions to be used. Defaults to 100.
        seed: int, seed to be used for random number generation. Defaults to None.
        
    Returns:
        swd_list: lists of float, list of swd values for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    values_list: np.ndarray = np.zeros(niter)
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if dist_1_symb is None:
            dist_1_num = dist_1_num[:nsamples,:]
        if dist_2_symb is None:
            dist_2_num = dist_2_num[:nsamples,:]
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar = tqdm(total = niter, desc = "Iterations")
        
    if verbose:
        print("Starting SWD metric calculation...")
        start: float = timer()
    
    if dist_1_symb is None and dist_2_symb is None:
        for k in range(niter):
            dist_1_k: tf.Tensor = dist_1_num[k*batch_size:(k+1)*batch_size,:]
            dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_k.dtype)
            directions /= tf.norm(directions, axis=1)[:, None]
            values_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_k, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
            if pbar:
                pbar.update(1) # type: ignore
    elif dist_1_symb is None and dist_2_symb is not None:
        for k in range(niter):
            dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
            dist_2_k = dist_2_symb.sample(batch_size)
            directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_k.dtype)
            directions /= tf.norm(directions, axis=1)[:, None]
            values_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_k, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
            if pbar:
                pbar.update(1) # type: ignore
    elif dist_1_symb is not None and dist_2_symb is None:
        for k in range(niter):
            dist_1_k = dist_1_symb.sample(batch_size)
            dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_k.dtype)
            directions /= tf.norm(directions, axis=1)[:, None]
            values_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_k, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
            if pbar:
                pbar.update(1) # type: ignore
    elif dist_1_symb is not None and dist_2_symb is not None:
        for k in range(niter):
            dist_1_k = dist_1_symb.sample(batch_size)
            dist_2_k = dist_2_symb.sample(batch_size)
            directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_k.dtype)
            directions /= tf.norm(directions, axis=1)[:, None]
            values_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_k, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
            if pbar:
                pbar.update(1) # type: ignore
                
    if verbose:
        end: float = timer()
        print("SWD metric calculation completed in {:.2f} seconds".format(end-start))

    return values_list.tolist()


def SWD_1_small(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                niter: int = 10,
                batch_size: int = 100000,
                nslices: int = 100,
                seed: Optional[int] = None,
                verbose: bool = False,
                progress_bar: bool = False
               ) -> List[float]:
    """
    Compute the sliced Wasserstein distance between two multi-dimensional distribution as the average over slices of the value of the 1D Wasserstein distances 
    between sets of points obtained projecting data along nslices random directions.
    The SWD value is used as a test statistic whose distribution is estimated by computing the quantity niter times. The niter list of SWD test statistic is returned.
    There are three cases:
        1. Both distributions are symbolic (tfp.distributions.Distribution). In this case, batch_size (input value) points dist_1_j, dist_2_j are sampled at each iteration.
        2. Both distributions are numerical (np.ndarray). In this case, if distributions have different number of samples, then the minimum number of samples is used. 
           Data are then split in niter batches dist_1_j, dist_2_j of size batch_size = int(nsamples/niter).
        3. One distribution is symbolic and the other is numerical. In this case, batch_size is set to int(nsamples/niter) of the numerical distribution. Data are then split in niter batches dist_1_j, dist_2_j, one taken from the numerical
           distribution and the other one sampled from the symbolic one.
    The swd_list is then computed over all pairs of batches dist_1_j, dist_2_j and the lists are returned.
    
    NOTE: The functions SWD_test_1 and SWD_test_2 differ in the way batches are computed and compared. For instance, for niter=10 and nsamples=100000, SWD_test_1 generates 10 pairs of independent batches 
    of 10000 samples each and performs the tests pairwise (total of 10 tests with 10000 points), while SWD_test_2 generates int(np.ceil(np.sqrt(niter)))=3 pairs of independent batches of int(100000/3)
    samples each and performs the tests for all pairs combinations (9 tests with int(100000/3) points). So the number of tests is always (almost) equal to niter, but SWD_test_1 performs independent 
    tests with less points, while SWD_test_2 performs non-independent tests with more points.
    
    Args:
        dist_1: np.ndarray or tfp.distributions.Distribution, the first distribution to be compared.
        dist_2: np.ndarray or tfp.distributions.Distribution, the second distribution to be compared.
        niter: int, number of iterations to be performed. Defaults to 10.
        batch_size: int, number of samples to be used in each iteration. Only used for numerical distributions. Defaults to 100000.
        nslices: int, number of random directions to be used. Defaults to 100.
        seed: int, seed to be used for random number generation. Defaults to None.
        
    Returns:
        swd_list: lists of float, list of swd values for each iteration.
    """ 
    dist_1_symb: tfp.distributions.Distribution
    dist_2_symb: tfp.distributions.Distribution
    dist_1_num: tf.Tensor
    dist_2_num: tf.Tensor
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    dtype_1: tf.DType
    dtype_2: tf.DType
    dtype: tf.DType
    values_list: np.ndarray = np.zeros(niter)
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if dist_1_symb is None:
        dist_1_num = dist_1_num[:nsamples,:]
    else:
        dist_1_num = dist_1_symb.sample(batch_size*niter)
    if dist_2_symb is None:
        dist_2_num = dist_2_num[:nsamples,:]
    else:
        dist_2_num = dist_2_symb.sample(batch_size*niter)
    if nsamples is not None:
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar = tqdm(total = niter, desc = "Iterations")
        
    if verbose:
        print("Starting SWD metric calculation...")
        start: float = timer()
    
    for k in range(niter):
        dist_1_k: tf.Tensor = dist_1_num[k*batch_size:(k+1)*batch_size,:]
        dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
        directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_k.dtype)
        directions /= tf.norm(directions, axis=1)[:, None]
        values_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_k, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
        if pbar:
            pbar.update(1)
            
    if verbose:
        end: float = timer()
        print("SWD metric calculation completed in {:.2f} seconds".format(end-start))

    return values_list.tolist()


def SWD_1(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
          dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
          niter: int = 10,
          batch_size: int = 100000,
          nslices: int = 100,
          seed: Optional[int] = None,
          verbose: bool = False,
          progress_bar: bool = False
         ) -> List[float]:
    _, _, ndims, _, _ = __parse_input_dist(dist_1_input, verbose = verbose)
    nn: int = batch_size*niter*ndims
    if nn < 1e7:
        try:
            return SWD_1_small(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
        except:
            return SWD_1_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
    else:
        return SWD_1_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)


def SWD_2_large(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                niter: int = 10,
                batch_size: int = 100000,
                nslices: int = 100,
                seed: Optional[int] = None,
                verbose: bool = False,
                progress_bar: bool = False
               ) -> List[float]:
    """
    Compute the sliced Wasserstein distance between two sets of points using nslices random directions and the p-th Wasserstein distance.
    The distance is computed for niter times and for nslices directions and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in np.ceil(np.sqrt(niter)) batches dist_1_j, dist_2_k of size batch_size=int(nsamples/np.ceil(np.sqrt(niter))) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_k.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        niter (int, optional): Number of iterations to be performed. Defaults to 100.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/niter
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    values_list: np.ndarray = np.zeros(niter)
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if dist_1_symb is None:
            dist_1_num = dist_1_num[:nsamples,:]
        if dist_2_symb is None:
            dist_2_num = dist_2_num[:nsamples,:]
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total = niter**2, desc = "Iterations")
        
    if verbose:
        print("Starting SWD metric calculation...")
        start: float = timer()
    
    l: int = 0
    if dist_1_symb is None and dist_2_symb is None:
        for j in range(niter):
            dist_1_j: tf.Tensor = dist_1_num[j*batch_size:(j+1)*batch_size,:]
            for k in range(niter):
                dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
                directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_j.dtype)
                directions /= tf.norm(directions, axis=1)[:, None]
                values_list[l] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_j, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is None and dist_2_symb is not None:
        for j in range(niter):
            dist_1_j = dist_1_num[j*batch_size:(j+1)*batch_size,:]
            for k in range(niter):
                dist_2_k = dist_2_symb.sample(batch_size)
                directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_j.dtype)
                directions /= tf.norm(directions, axis=1)[:, None]
                values_list[l] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_j, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is None:
        for j in range(niter):
            dist_1_j = dist_1_symb.sample(batch_size)
            for k in range(niter):
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]  
                directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_j.dtype)
                directions /= tf.norm(directions, axis=1)[:, None]
                values_list[l] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_j, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is not None:
        for j in range(niter):
            dist_1_j = dist_1_symb.sample(batch_size)
            for k in range(niter):
                dist_2_k = dist_2_symb.sample(batch_size)
                directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_j.dtype)
                directions /= tf.norm(directions, axis=1)[:, None]
                values_list[l] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_j, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
                l += 1
                if pbar:
                    pbar.update(1)
                    
    if verbose:
        end: float = timer()
        print("SWD metric calculation completed in {:.2f} seconds".format(end-start))

    return values_list.tolist()
        

def SWD_2_small(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                niter: int = 10,
                batch_size: int = 100000,
                nslices: int = 100,
                seed: Optional[int] = None,
                verbose: bool = False,
                progress_bar: bool = False
               ) -> List[float]:
    """
    Compute the sliced Wasserstein distance between two sets of points using nslices random directions and the p-th Wasserstein distance.
    The distance is computed for niter times and for nslices directions and the mean and std of the p-values are returned.
    In the case of numerical distributions, data are split in np.ceil(np.sqrt(niter)) batches dist_1_j, dist_2_k of size batch_size=int(nsamples/np.ceil(np.sqrt(niter))) and the mean and std are computed over all pairs of batches dist_1_j, dist_2_k.
    Args:
        dist_1 (numpy array or distribution): The first distribution to be compared
        dist_2 (numpy array or distribution): The second distribution to be compared
        niter (int, optional): Number of iterations to be performed. Defaults to 100.
        batch_size (int, optional): Number of samples to be used in each iteration. Only used if num is true. Defaults to 100000.
    Returns:
        [float,float]: Mean and std of the p-values obtained from the KS tests
    """
    # If an array of the input is an array, then the input batch_size is ignored and batch_size is set to nsamples/niter
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    values_list: np.ndarray = np.zeros(niter)
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if dist_1_symb is None:
        dist_1_num = dist_1_num[:nsamples,:]
    else:
        dist_1_num = dist_1_symb.sample(batch_size*niter)
    if dist_2_symb is None:
        dist_2_num = dist_2_num[:nsamples,:]
    else:
        dist_2_num = dist_2_symb.sample(batch_size*niter)
    if nsamples is not None:
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total = niter**2, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting SWD metric calculation...")
        start: float = timer()
    
    l: int = 0  
    for j in range(niter):
        dist_1_j: tf.Tensor = dist_1_num[j*batch_size:(j+1)*batch_size,:]
        for k in range(niter):
            dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_j.dtype)
            directions /= tf.norm(directions, axis=1)[:, None]
            values_list[l] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_j, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
            l += 1
            if pbar:
                pbar.update(1)
                
    if verbose:
        end: float = timer()
        print("SWD metric calculation completed in {:.2f} seconds".format(end-start))

    return values_list.tolist()
        

def SWD_2(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
          dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
          niter: int = 10,
          batch_size: int = 100000,
          nslices: int = 100,
          seed: Optional[int] = None,
          verbose: bool = False,
          progress_bar: bool = False
         ) -> List[float]:
    _, _, ndims, _, _ = __parse_input_dist(dist_1_input, verbose = verbose)
    nn: int = batch_size*niter*ndims
    if nn < 1e7:
        try:
            return SWD_2_small(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
        except:
            return SWD_2_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)
    else:
        return SWD_2_large(dist_1_input, dist_2_input, niter, batch_size, verbose, progress_bar)


def ComputeMetrics_1_large(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                           dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                           niter: int = 10,
                           batch_size: int = 100000,
                           nslices: int = 100,
                           seed: Optional[int] = None,
                           verbose: bool = False,
                           progress_bar: bool = False
                          ) -> Tuple[List[float],...]:
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    ks_metric_list: np.ndarray = np.zeros(niter)
    ks_pvalue_list: np.ndarray = np.zeros(niter)
    ad_metric_list: np.ndarray = np.zeros(niter)
    ad_pvalue_list: np.ndarray = np.zeros(niter)
    fn_list: np.ndarray = np.zeros(niter)
    wd_list: np.ndarray = np.zeros(niter)
    swd_list: np.ndarray = np.zeros(niter)
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if dist_1_symb is None:
            dist_1_num = dist_1_num[:nsamples,:]
        if dist_2_symb is None:
            dist_2_num = dist_2_num[:nsamples,:]
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar = tqdm(total = niter, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting metrics calculation...")
        start: float = timer()
    
    if dist_1_symb is None and dist_2_symb is None:
        for k in range(niter):
            dist_1_k: tf.Tensor = dist_1_num[k*batch_size:(k+1)*batch_size,:]
            dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            [ks_metric_list[k], ks_pvalue_list[k]] = np.mean([ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
            test = [anderson_ksamp([dist_1_k[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
            [ad_metric_list[k], ad_pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
            dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_k, sample_axis=0, event_axis=-1))
            dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
            fn_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
            wd_list[k] = np.mean([wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
            directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_k.dtype)
            directions /= tf.norm(directions, axis=1)[:, None]
            swd_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_k, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
            if pbar:
                pbar.update(1)
    elif dist_1_symb is None and dist_2_symb is not None:
        for k in range(niter):
            dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
            dist_2_k = dist_2_symb.sample(batch_size)
            [ks_metric_list[k], ks_pvalue_list[k]] = np.mean([ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
            test = [anderson_ksamp([dist_1_k[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
            [ad_metric_list[k], ad_pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
            dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_k, sample_axis=0, event_axis=-1))
            dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
            fn_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
            wd_list[k] = np.mean([wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
            directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_k.dtype)
            directions /= tf.norm(directions, axis=1)[:, None]
            swd_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_k, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
            if pbar:
                pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is None:
        for k in range(niter):
            dist_1_k = dist_1_symb.sample(batch_size)
            dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            [ks_metric_list[k], ks_pvalue_list[k]] = np.mean([ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
            test = [anderson_ksamp([dist_1_k[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
            [ad_metric_list[k], ad_pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
            dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_k, sample_axis=0, event_axis=-1))
            dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
            fn_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
            wd_list[k] = np.mean([wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
            directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_k.dtype)
            directions /= tf.norm(directions, axis=1)[:, None]
            swd_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_k, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
            if pbar:
                pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is not None:
        for k in range(niter):
            dist_1_k = dist_1_symb.sample(batch_size)
            dist_2_k = dist_2_symb.sample(batch_size)
            [ks_metric_list[k], ks_pvalue_list[k]] = np.mean([ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
            test = [anderson_ksamp([dist_1_k[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
            [ad_metric_list[k], ad_pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
            dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_k, sample_axis=0, event_axis=-1))
            dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
            fn_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
            wd_list[k] = np.mean([wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
            directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_k.dtype)
            directions /= tf.norm(directions, axis=1)[:, None]
            swd_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_k, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
            if pbar:
                pbar.update(1)
                
    if verbose:
        end: float = timer()
        print("Metrics calculation completed in {:.2f} seconds".format(end-start))
    
    return ks_metric_list.tolist(), ks_pvalue_list.tolist(), ad_metric_list.tolist(), ad_pvalue_list.tolist(), fn_list.tolist(), wd_list.tolist(), swd_list.tolist()


def ComputeMetrics_1_small(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                           dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                           niter: int = 10,
                           batch_size: int = 100000,
                           nslices: int = 100,
                           seed: Optional[int] = None,
                           verbose: bool = False,
                           progress_bar: bool = False
                          ) -> Tuple[List[float],...]:
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    ks_metric_list: np.ndarray = np.zeros(niter)
    ks_pvalue_list: np.ndarray = np.zeros(niter)
    ad_metric_list: np.ndarray = np.zeros(niter)
    ad_pvalue_list: np.ndarray = np.zeros(niter)
    fn_list: np.ndarray = np.zeros(niter)
    wd_list: np.ndarray = np.zeros(niter)
    swd_list: np.ndarray = np.zeros(niter)
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if dist_1_symb is None:
        dist_1_num = dist_1_num[:nsamples,:]
    else:
        dist_1_num = dist_1_symb.sample(batch_size*niter)
    if dist_2_symb is None:
        dist_2_num = dist_2_num[:nsamples,:]
    else:
        dist_2_num = dist_2_symb.sample(batch_size*niter)
    if nsamples is not None:
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar = tqdm(total = niter, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting metrics calculation...")
        start: float = timer()
    
    for k in range(niter):
        dist_1_k: tf.Tensor = dist_1_num[k*batch_size:(k+1)*batch_size,:]
        dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
        [ks_metric_list[k], ks_pvalue_list[k]] = np.mean([ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
        test = [anderson_ksamp([dist_1_k[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
        [ad_metric_list[k], ad_pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
        dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_k, sample_axis=0, event_axis=-1))
        dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
        fn_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
        wd_list[k] = np.mean([wasserstein_distance(dist_1_k[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
        directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_k.dtype)
        directions /= tf.norm(directions, axis=1)[:, None]
        swd_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_k, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
        if pbar:
            pbar.update(1)
            
    if verbose:
        end: float = timer()
        print("Metrics calculation completed in {:.2f} seconds".format(end-start))
    
    return ks_metric_list.tolist(), ks_pvalue_list.tolist(), ad_metric_list.tolist(), ad_pvalue_list.tolist(), fn_list.tolist(), wd_list.tolist(), swd_list.tolist()


def ComputeMetrics_1(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                     dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                     niter: int = 10,
                     batch_size: int = 100000,
                     nslices: int = 100,
                     seed: Optional[int] = None,
                     verbose: bool = False,
                     progress_bar: bool = False
                    ) -> Tuple[List[float],...]:
    _, _, ndims, _, _ = __parse_input_dist(dist_1_input, verbose = verbose)
    nn: int = batch_size*niter*ndims
    if nn < 1e7:
        try:
            return ComputeMetrics_1_small(dist_1_input, dist_2_input, niter, batch_size, nslices, seed, verbose, progress_bar)
        except:
            return ComputeMetrics_1_large(dist_1_input, dist_2_input, niter, batch_size, nslices, seed, verbose, progress_bar)
    else:
        return ComputeMetrics_1_large(dist_1_input, dist_2_input, niter, batch_size, nslices, seed, verbose, progress_bar)


def ComputeMetrics_2_large(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                           dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                           niter: int = 10,
                           batch_size: int = 100000,
                           nslices: int = 100,
                           seed: Optional[int] = None,
                           verbose: bool = False,
                           progress_bar: bool = False
                          ) -> Tuple[List[float],...]:
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    ks_metric_list: np.ndarray = np.zeros(niter)
    ks_pvalue_list: np.ndarray = np.zeros(niter)
    ad_metric_list: np.ndarray = np.zeros(niter)
    ad_pvalue_list: np.ndarray = np.zeros(niter)
    fn_list: np.ndarray = np.zeros(niter)
    wd_list: np.ndarray = np.zeros(niter)
    swd_list: np.ndarray = np.zeros(niter)
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if nsamples is not None:
        if dist_1_symb is None:
            dist_1_num = dist_1_num[:nsamples,:]
        if dist_2_symb is None:
            dist_2_num = dist_2_num[:nsamples,:]
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total = niter**2, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting metrics calculation...")
        start: float = timer()
    
    l: int = 0
    if dist_1_symb is None and dist_2_symb is None:
        for j in range(niter):
            dist_1_j: tf.Tensor = dist_1_num[j*batch_size:(j+1)*batch_size,:]
            for k in range(niter):
                dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
                [ks_metric_list[k], ks_pvalue_list[k]] = np.mean([ks_2samp(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
                test = [anderson_ksamp([dist_1_j[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
                [ad_metric_list[k], ad_pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
                dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_j, sample_axis=0, event_axis=-1))
                dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
                fn_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
                wd_list[k] = np.mean([wasserstein_distance(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
                directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_j.dtype)
                directions /= tf.norm(directions, axis=1)[:, None]
                swd_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_j, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is None and dist_2_symb is not None:
        for j in range(niter):
            dist_1_j = dist_1_num[j*batch_size:(j+1)*batch_size,:]
            for k in range(niter):
                dist_2_k = dist_2_symb.sample(batch_size)
                [ks_metric_list[k], ks_pvalue_list[k]] = np.mean([ks_2samp(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
                test = [anderson_ksamp([dist_1_j[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
                [ad_metric_list[k], ad_pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
                dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_j, sample_axis=0, event_axis=-1))
                dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
                fn_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
                wd_list[k] = np.mean([wasserstein_distance(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
                directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_j.dtype)
                directions /= tf.norm(directions, axis=1)[:, None]
                swd_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_j, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is None:
        for j in range(niter):
            dist_1_j = dist_1_symb.sample(batch_size)
            for k in range(niter):
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]    
                [ks_metric_list[k], ks_pvalue_list[k]] = np.mean([ks_2samp(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
                test = [anderson_ksamp([dist_1_j[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
                [ad_metric_list[k], ad_pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
                dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_j, sample_axis=0, event_axis=-1))
                dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
                fn_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
                wd_list[k] = np.mean([wasserstein_distance(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
                directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_j.dtype)
                directions /= tf.norm(directions, axis=1)[:, None]
                swd_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_j, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
                l += 1
                if pbar:
                    pbar.update(1)
    elif dist_1_symb is not None and dist_2_symb is not None:
        for j in range(niter):
            dist_1_j = dist_1_symb.sample(batch_size)
            for k in range(niter):
                dist_2_k = dist_2_symb.sample(batch_size)
                [ks_metric_list[k], ks_pvalue_list[k]] = np.mean([ks_2samp(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
                test = [anderson_ksamp([dist_1_j[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
                [ad_metric_list[k], ad_pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
                dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_j, sample_axis=0, event_axis=-1))
                dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
                fn_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
                wd_list[k] = np.mean([wasserstein_distance(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
                directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_j.dtype)
                directions /= tf.norm(directions, axis=1)[:, None]
                swd_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_j, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
                l += 1
                if pbar:
                    pbar.update(1)
                    
    if verbose:
        end: float = timer()
        print("Metrics calculation completed in {:.2f} seconds".format(end-start))
    
    return ks_metric_list.tolist(), ks_pvalue_list.tolist(), ad_metric_list.tolist(), ad_pvalue_list.tolist(), fn_list.tolist(), wd_list.tolist(), swd_list.tolist()


def ComputeMetrics_2_small(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                           dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                           niter: int = 10,
                           batch_size: int = 100000,
                           nslices: int = 100,
                           seed: Optional[int] = None,
                           verbose: bool = False,
                           progress_bar: bool = False  
                          ) -> Tuple[List[float],...]:
    ndims_1: int
    ndims_2: int
    nsamples_1: Optional[int]
    nsamples_2: Optional[int]
    nsamples: Optional[int] = None
    ks_metric_list: np.ndarray = np.zeros(niter)
    ks_pvalue_list: np.ndarray = np.zeros(niter)
    ad_metric_list: np.ndarray = np.zeros(niter)
    ad_pvalue_list: np.ndarray = np.zeros(niter)
    fn_list: np.ndarray = np.zeros(niter)
    wd_list: np.ndarray = np.zeros(niter)
    swd_list: np.ndarray = np.zeros(niter)
    
    niter = int(np.ceil(np.sqrt(niter)))
    
    dist_1_symb, dist_1_num, ndims_1, nsamples_1, dtype_1 = __parse_input_dist(dist_1_input, verbose = verbose)
    dist_2_symb, dist_2_num, ndims_2, nsamples_2, dtype_2 = __parse_input_dist(dist_2_input, verbose = verbose)
    
    if ndims_1 != ndims_2:
        raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
    ndims: int = ndims_1
    
    if nsamples_1 is not None and nsamples_2 is not None:
        nsamples = min(nsamples_1, nsamples_2)
    elif nsamples_1 is not None:
        nsamples = nsamples_1
    elif nsamples_2 is not None:
        nsamples = nsamples_2
    if dist_1_symb is None:
        dist_1_num = dist_1_num[:nsamples,:]
    else:
        dist_1_num = dist_1_symb.sample(batch_size*niter)
    if dist_2_symb is None:
        dist_2_num = dist_2_num[:nsamples,:]
    else:
        dist_2_num = dist_2_symb.sample(batch_size*niter)
    if nsamples is not None:
        batch_size = nsamples // niter
        if batch_size == 0:
            raise ValueError("nsamples must be greater than niter when at least one distribution is numerical")
        
    if progress_bar:
        pbar: Optional[tqdm] = tqdm(total = niter**2, desc = "Iterations")
    else:
        pbar = None
        
    if verbose:
        print("Starting metrics calculation...")
        start: float = timer()
    
    l: int = 0
    for j in range(niter):
        dist_1_j: tf.Tensor = dist_1_num[j*batch_size:(j+1)*batch_size,:]
        for k in range(niter):
            dist_2_k: tf.Tensor = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            [ks_metric_list[k], ks_pvalue_list[k]] = np.mean([ks_2samp(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)],axis=0).tolist()
            test = [anderson_ksamp([dist_1_j[:,dim], dist_2_k[:,dim]]) for dim in range(ndims)]
            [ad_metric_list[k], ad_pvalue_list[k]] = np.array([[p[0],p[2]] for p in test]).mean(axis=0)
            dist_1_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_1_j, sample_axis=0, event_axis=-1))
            dist_2_corr = correlation_from_covariance_tf(tfp.stats.covariance(dist_2_k, sample_axis=0, event_axis=-1))
            fn_list[k] = float(tf.norm(dist_1_corr - dist_2_corr).numpy())
            wd_list[k] = np.mean([wasserstein_distance(dist_1_j[:,dim], dist_2_k[:,dim]) for dim in range(ndims)])
            directions = tf.random.normal(shape=(nslices, ndims),dtype=dist_1_j.dtype)
            directions /= tf.norm(directions, axis=1)[:, None]
            swd_list[k] = np.mean([wasserstein_distance(tf.linalg.matvec(dist_1_j, direction), tf.linalg.matvec(dist_2_k, direction)) for direction in directions])
            l += 1
            if pbar:
                pbar.update(1)
                
    if verbose:
        end: float = timer()
        print("Metrics calculation completed in {:.2f} seconds".format(end-start))
    
    return ks_metric_list.tolist(), ks_pvalue_list.tolist(), ad_metric_list.tolist(), ad_pvalue_list.tolist(), fn_list.tolist(), wd_list.tolist(), swd_list.tolist()


def ComputeMetrics_2(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                     dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                     niter: int = 10,
                     batch_size: int = 100000,
                     nslices: int = 100,
                     seed: Optional[int] = None,
                     verbose: bool = False,
                     progress_bar: bool = False
                    ) -> Tuple[List[float],...]:
    _, _, ndims, _, _ = __parse_input_dist(dist_1_input, verbose = verbose)
    nn: int = batch_size*niter*ndims
    if nn < 1e7:
        try:
            return ComputeMetrics_2_small(dist_1_input, dist_2_input, niter, batch_size, nslices, seed, verbose, progress_bar)
        except:
            return ComputeMetrics_2_large(dist_1_input, dist_2_input, niter, batch_size, nslices, seed, verbose, progress_bar)
    else:
        return ComputeMetrics_2_large(dist_1_input, dist_2_input, niter, batch_size, nslices, seed, verbose, progress_bar)


def ComputeMetrics(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                   dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                   niter: int = 10,
                   batch_size: int = 100000,
                   nslices: int = 100,
                   seed: Optional[int] = None,
                   verbose: bool = False,
                   progress_bar: bool = False
                  ) -> Tuple[List[float],...]:
    return ComputeMetrics_1(dist_1_input, dist_2_input, niter, batch_size, nslices, seed, verbose, progress_bar)


def ComputeMetrics_sequential(dist_1_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                              dist_2_input: Union[np.ndarray, tfp.distributions.Distribution, tf.Tensor],
                              niter: int = 10,
                              batch_size: int = 100000,
                              nslices: int = 100,
                              seed: Optional[int] = None,
                              verbose: bool = False,
                              progress_bar: bool = False
                             ) -> Tuple[List[float],...]:
    """
    Function that computes the metrics. The following metrics are computed:
        - Mean of test statistics and p-values of 1D KS tests
        - Mean of test statistics and p-values of 1D AD tests
        - Values of FN metric
        - Mean of values of 1D WD
        - SWD values
    """
    ks_metric_list, ks_pvalue_list = KS_test_1_large(dist_1_input, dist_2_input, niter = niter, batch_size = batch_size, verbose = verbose, progress_bar = progress_bar)
    ad_metric_list, ad_pvalue_list = AD_test_1_large(dist_1_input, dist_2_input, niter = niter, batch_size = batch_size, verbose = verbose, progress_bar = progress_bar)
    fn_list = FN_1_large(dist_1_input, dist_2_input, niter = niter, batch_size = batch_size, verbose = verbose, progress_bar = progress_bar)
    wd_list = WD_1_large(dist_1_input, dist_2_input, niter = niter, batch_size = batch_size, verbose = verbose, progress_bar = progress_bar)
    swd_list = SWD_1_large(dist_1_input, dist_2_input, niter = niter, batch_size = batch_size, nslices = nslices, seed = seed, verbose = verbose, progress_bar = progress_bar)
    return ks_metric_list, ks_pvalue_list, ad_metric_list, ad_pvalue_list, fn_list, wd_list, swd_list
