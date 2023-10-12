__all__ = ["FNMetric"]

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import traceback
from datetime import datetime
from timeit import default_timer as timer
from tqdm import tqdm # type: ignore
from .utils import reset_random_seeds
from .utils import conditional_print
from .utils import conditional_tf_print
from .utils import generate_and_clean_data
from .utils import NumpyDistribution
from .base import TwoSampleTestInputs
from .base import TwoSampleTestBase
from .base import TwoSampleTestResult
from .base import TwoSampleTestResults

from typing import Tuple, Union, Optional, Type, Dict, Any, List
from .utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType

# dist_1_cov = np.cov(dist_1_k,bias=True,rowvar=False)
# dist_1_corr=correlation_from_covariance(dist_1_cov)
# dist_2_cov = np.cov(dist_2_k,bias=True,rowvar=False)
# dist_2_corr=correlation_from_covariance(dist_2_cov)    
# matrix_sum=dist_1_corr-dist_2_corr
# frob_norm=np.linalg.norm(matrix_sum, ord='fro')
# fn_list.append(frob_norm)

def correlation_from_covariance_np(covariance: np.ndarray) -> np.ndarray:
    """
    """
    stddev: np.ndarray = np.sqrt(np.diag(covariance))
    correlation: np.ndarray = covariance / np.outer(stddev, stddev)
    correlation = np.where(np.equal(covariance, 0), 0, correlation) 
    return correlation


def correlation_from_covariance_tf(covariance: tf.Tensor) -> tf.Tensor:
    """
    """
    stddev = tf.sqrt(tf.linalg.diag_part(covariance))
    correlation = covariance / (stddev[:, None] * stddev[None, :])
    correlation = tf.where(tf.equal(covariance, 0), tf.constant(0, dtype=correlation.dtype), correlation)
    return correlation

def fn_2samp_np(data1: np.ndarray,
                data2: np.ndarray
               ) -> np.ndarray:
    """
    """
    dist_1_cov = np.cov(data1, bias=True, rowvar=False)
    dist_1_corr = correlation_from_covariance_np(dist_1_cov)
    dist_2_cov = np.cov(data2, bias=True, rowvar=False)
    dist_2_corr = correlation_from_covariance_np(dist_2_cov)
    matrix_sum = dist_1_corr - dist_2_corr
    frob_norm = np.linalg.norm(matrix_sum, ord='fro')
    return frob_norm # type: ignore

@tf.function(experimental_compile=True)
def fn_2samp_tf(data1: tf.Tensor, 
                data2: tf.Tensor
               ) -> tf.Tensor:
    """
    """
    dist_1_cov = tfp.stats.covariance(data1, sample_axis=0, event_axis=-1)
    dist_1_corr = correlation_from_covariance_tf(dist_1_cov)
    dist_2_cov = tfp.stats.covariance(data2, sample_axis=0, event_axis=-1)
    dist_2_corr = correlation_from_covariance_tf(dist_2_cov)    
    matrix_sum = tf.subtract(dist_1_corr, dist_2_corr)
    frob_norm = tf.norm(matrix_sum, ord='fro', axis=[-2,-1])
    return frob_norm

class FNMetric(TwoSampleTestBase):
    """
    Class for computing the Frobenius norm between the correlation matrices of the two samples.
    It inherits from the TwoSampleTestBase class.
    The FN is computed by projecting the samples onto random directions, 
    computing the Wasserstein distance between the projections, and
    then taking the mean and standard deviation of the Wasserstein distances.
    The Frobenius norm can be computed using either numpy or tensorflow.
    The scipy implementation is used for the numpy backend.
    A custom tensorflow implementation is used for the tensorflow backend.
    The tensorflow implementation is faster than the scipy implementation, especially for large sample sizes,
    number of projections, and number of iterations.
    
    Parameters:
    ----------
    data_input: TwoSampleTestInputs
        Object containing the inputs for the two-sample test.

    progress_bar: bool, optional, default = False
        If True, display a progress bar. The progress bar is automatically disabled when running tensorflow functions.
        
    verbose: bool, optional, default = False
        If True, print additional information.

    Attributes:
    ----------
    Inputs: TwoSampleTestInputs object
        Object containing the inputs for the two-sample test.

    Results: TwoSampleTestResults object
        Object containing the results of the two-sample test.

    start: float
        Time when the two-sample test calculation started.

    end: float
        Time when the two-sample test calculation ended.

    pbar: tqdm
        Progress bar object.
        
    Methods:
    -------
    compute() -> None
        Function that computes the Frobenius norm between the correlation matrices of the two samples
        selecting among the Test_np and Test_tf methods depending on the value of the use_tf attribute.
        
    Test_np() -> None
        Function that computes the Frobenius norm between the correlation matrices 
        of the two samples using numpy functions.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
        
    Test_tf() -> None
        Function that computes the Frobenius norm between the correlation matrices
        of the two samples using tensorflow functions.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
        
    Examples:
    --------
    
    .. code-block:: python
    
        import numpy as np
        import tensorflow as tf
        import tensorflow_probability as tfp
        import GenerativeModelsMetrics as GMetrics

        # Set random seed
        seed = 0
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Define inputs
        nsamples = 1000000
        ndims = 2
        dtype = tf.float32
        ndims = 100
        eps = 0.1
        dist_1_symb = tfp.distributions.Normal(loc=np.full(ndims,0.), scale=np.full(ndims,1.))
        dist_2_symb = tfp.distributions.Normal(loc=np.random.uniform(-eps, eps, ndims), scale=np.random.uniform(1-eps, 1+eps, ndims))
        dist_1_num = tf.cast(dist_1_symb.sample(nsamples),tf.float32)
        dist_2_num = tf.cast(dist_2_symb.sample(nsamples),tf.float32)
        data_input = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1_num,
                                                  dist_2_input = dist_2_num,
                                                  niter = 100,
                                                  batch_size = 10000,
                                                  dtype_input = tf.float64,
                                                  seed_input = 0,
                                                  use_tf = True,
                                                  verbose = True)

        # Compute FN metric
        FN_metric = GMetrics.FNMetric(data_input = data_input, 
                                        progress_bar = True, 
                                        verbose = True)
        FN_metric.compute()
        FN_metric.Results[0].result_value
        
        >> {'metric_list': [...]}
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 progress_bar: bool = False,
                 verbose: bool = False
                ) -> None:
        # From base class
        self._Inputs: TwoSampleTestInputs
        self._progress_bar: bool
        self._verbose: bool
        self._start: float
        self._end: float
        self._pbar: tqdm
        self._Results: TwoSampleTestResults
    
        super().__init__(data_input = data_input, 
                         progress_bar = progress_bar,
                         verbose = verbose)
            
    def compute(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the Frobenius norm between the correlation matrices of the two samples
        selecting among the Test_np and Test_tf methods depending on the value of the use_tf attribute.
        
        Parameters:
        ----------
        max_vectorize: int, optional, default = 100
            Maximum number of samples that can be processed by the tensorflow backend.
            If None, the total number of samples is not checked.

        Returns:
        -------
        None
        """
        if self.use_tf:
            self.Test_tf(max_vectorize = max_vectorize)
        else:
            self.Test_np()
    
    def Test_np(self) -> None:
        """
        Function that computes the Frobenius norm between the correlation matrices
        The number of random directions used for the projection is given by nslices.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        # Set alias for inputs
        if isinstance(self.Inputs.dist_1_num, np.ndarray):
            dist_1_num: DataTypeNP = self.Inputs.dist_1_num
        else:
            dist_1_num = self.Inputs.dist_1_num.numpy()
        if isinstance(self.Inputs.dist_2_num, np.ndarray):
            dist_2_num: DataTypeNP = self.Inputs.dist_2_num
        else:
            dist_2_num = self.Inputs.dist_2_num.numpy()
        dist_1_symb: DistType = self.Inputs.dist_1_symb
        dist_2_symb: DistType = self.Inputs.dist_2_symb
        ndims: int = self.Inputs.ndims
        niter: int
        batch_size: int
        niter, batch_size = self.get_niter_batch_size_np() # type: ignore
        if isinstance(self.Inputs.dtype, tf.DType):
            dtype: Union[type, np.dtype] = self.Inputs.dtype.as_numpy_dtype
        else:
            dtype = self.Inputs.dtype
        seed: int = self.Inputs.seed
        dist_1_k: DataTypeNP
        dist_2_k: DataTypeNP
        
        # Utility functions
        def start_calculation() -> None:
            conditional_print(self.verbose, "\n------------------------------------------")
            conditional_print(self.verbose, "Starting FN metric calculation...")
            conditional_print(self.verbose, "niter = {}" .format(niter))
            conditional_print(self.verbose, "batch_size = {}" .format(batch_size))
            self._start = timer()
            
        def init_progress_bar() -> None:
            nonlocal niter
            if self.progress_bar:
                self.pbar = tqdm(total = niter, desc="Iterations")

        def update_progress_bar() -> None:
            if not self.pbar.disable:
                self.pbar.update(1)

        def close_progress_bar() -> None:
            if not self.pbar.disable:
                self.pbar.close()

        def end_calculation() -> None:
            self._end = timer()
            conditional_print(self.verbose, "Two-sample test calculation completed in "+str(self.end-self.start)+" seconds.")
        
        metric_list: List[DataTypeNP] = []

        start_calculation()
        init_progress_bar()
            
        reset_random_seeds(seed = seed)
        
        conditional_print(self.verbose, "Running numpy FN calculation...")
        for k in range(niter):
            if not np.shape(dist_1_num[0])[0] == 0 and not np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            elif not np.shape(dist_1_num[0])[0] == 0 and np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
                dist_2_k = np.array(dist_2_symb.sample(batch_size)).astype(dtype) # type: ignore
            elif np.shape(dist_1_num[0])[0] == 0 and not np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = np.array(dist_1_symb.sample(batch_size)).astype(dtype) # type: ignore
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            else:
                dist_1_k = np.array(dist_1_symb.sample(batch_size)).astype(dtype) # type: ignore
                dist_2_k = np.array(dist_2_symb.sample(batch_size)).astype(dtype) # type: ignore
            frob_norm = fn_2samp_np(dist_1_k, dist_2_k)
            metric_list.append(frob_norm) # type: ignore
            update_progress_bar()
        
        close_progress_bar()
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "FN Test_np"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy"}}
        result_value: Dict[str, DataTypeTF] = {"metric_list": np.array(metric_list)} # type: ignore
        result = TwoSampleTestResult(timestamp, test_name, parameters, result_value) # type: ignore
        self.Results.append(result)
        
    def Test_tf(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the Frobenuis norm between the correlation matrices 
        of the two samples using tensorflow functions.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The calculation is parallelized over max_vectorize (out of niter).
        The results are stored in the Results attribute.

        Parameters:
        ----------
        max_vectorize: int, optional, default = 100
            A maximum number of batch_size*max_vectorize samples per time are processed by the tensorflow backend.
            Given a value of max_vectorize, the niter FN calculations are split in chunks of max_vectorize.
            Each chunk is processed by the tensorflow backend in parallel. If ndims is larger than max_vectorize,
            the calculation is vectorized niter times over ndims.

        Returns:
        --------
        None
        """
        # Set alias for inputs
        if isinstance(self.Inputs.dist_1_num, np.ndarray):
            dist_1_num: tf.Tensor = tf.convert_to_tensor(self.Inputs.dist_1_num)
        else:
            dist_1_num = self.Inputs.dist_1_num # type: ignore
        if isinstance(self.Inputs.dist_2_num, np.ndarray):
            dist_2_num: tf.Tensor = tf.convert_to_tensor(self.Inputs.dist_2_num)
        else:
            dist_2_num = self.Inputs.dist_2_num # type: ignore
        if isinstance(self.Inputs.dist_1_symb, tfp.distributions.Distribution):
            dist_1_symb: tfp.distributions.Distribution = self.Inputs.dist_1_symb
        else:
            raise ValueError("dist_1_symb must be a tfp.distributions.Distribution object when use_tf is True.")
        if isinstance(self.Inputs.dist_2_symb, tfp.distributions.Distribution):
            dist_2_symb: tfp.distributions.Distribution = self.Inputs.dist_2_symb
        else:
            raise ValueError("dist_2_symb must be a tfp.distributions.Distribution object when use_tf is True.")
        ndims: int = self.Inputs.ndims
        niter: int
        batch_size: int
        niter, batch_size = [int(i) for i in self.get_niter_batch_size_tf()] # type: ignore
        dtype: tf.DType = tf.as_dtype(self.Inputs.dtype)
        seed: int = self.Inputs.seed
        
        # Utility functions
        def start_calculation() -> None:
            conditional_tf_print(self.verbose, "\n------------------------------------------")
            conditional_tf_print(self.verbose, "Starting FN metric calculation...")
            conditional_tf_print(self.verbose, "Running TF FN calculation...")
            conditional_tf_print(self.verbose, "niter =", niter)
            conditional_tf_print(self.verbose, "batch_size =", batch_size)
            self._start = timer()

        def end_calculation() -> None:
            self._end = timer()
            elapsed = self.end - self.start
            conditional_tf_print(self.verbose, "FN metric calculation completed in", str(elapsed), "seconds.")
                    
        def set_dist_num_from_symb(dist: DistTypeTF,
                                   nsamples: int,
                                   seed: int = 0
                                  ) -> tf.Tensor:
            nonlocal dtype
            #dist_num: tf.Tensor = tf.cast(dist.sample(nsamples, seed = int(seed)), dtype = dtype) # type: ignore
            dist_num: tf.Tensor = generate_and_clean_data(dist, nsamples, 100, dtype = self.Inputs.dtype, seed = int(seed), mirror_strategy = self.Inputs.mirror_strategy) # type: ignore
            return dist_num
        
        def return_dist_num(dist_num: tf.Tensor) -> tf.Tensor:
            return dist_num
            
        #@tf.function(reduce_retracing=True)
        def batched_test(start, end):
            conditional_tf_print(tf.logical_and(tf.logical_or(tf.math.logical_not(tf.equal(start,0)),tf.math.logical_not(tf.equal(end,niter))), self.verbose), "Iterating from", start, "to", end, "out of", niter, ".")
            seed_dist_1  = int(1e6)  # Seed for distribution 1
            seed_dist_2  = int(1e12)  # Seed for distribution 2

            dist_1_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_1_symb, nsamples = batch_size*(end-start), seed = seed_dist_1),
                                               false_fn = lambda: return_dist_num(dist_1_num[start*batch_size:end*batch_size, :])) # type: ignore
            dist_2_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_2_symb, nsamples = batch_size*(end-start), seed = seed_dist_2),
                                               false_fn = lambda: return_dist_num(dist_2_num[start*batch_size:end*batch_size, :])) # type: ignore

            dist_1_k = tf.reshape(dist_1_k, (end-start, batch_size, ndims))
            dist_2_k = tf.reshape(dist_2_k, (end-start, batch_size, ndims))

            # Define the loop body to vectorize over ndims*chunk_size
            def loop_body_vmap(idx):
                fron_norm = fn_2samp_tf(dist_1_k[idx, :, :], dist_2_k[idx, :, :]) # type: ignore
                fron_norm = tf.cast(fron_norm, dtype=dtype)
                return fron_norm

            # Vectorize over ndims*chunk_size
            frob_norm_list = tf.vectorized_map(loop_body_vmap, tf.range(end-start)) # type: ignore

            return frob_norm_list
        
        #@tf.function(reduce_retracing=True)
        def compute_test(max_vectorize: int = 100) -> tf.Tensor:
            # Check if numerical distributions are empty and print a warning if so
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_1_num tensor is empty. Batches will be generated 'on-the-fly' from dist_1_symb.") # type: ignore
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_2_num tensor is empty. Batches will be generated 'on-the-fly' from dist_2_symb.") # type: ignore
            
            # Ensure that max_vectorize is an integer larger than ndims
            max_vectorize = int(tf.cast(tf.maximum(max_vectorize, ndims),tf.int32)) # type: ignore

            # Compute the maximum number of iterations per chunk
            max_iter_per_chunk: int = int(tf.cast(tf.math.floor(max_vectorize / ndims), tf.int32)) # type: ignore
            
           # Compute the number of chunks
            nchunks: int = int(tf.cast(tf.math.ceil(niter / max_iter_per_chunk), tf.int32)) # type: ignore
            conditional_tf_print(tf.logical_and(self.verbose,tf.logical_not(tf.equal(nchunks,1))), "nchunks =", nchunks)

            res = tf.TensorArray(dtype, size = nchunks)

            def body(i, res):
                start = i * max_iter_per_chunk
                end = tf.minimum(start + max_iter_per_chunk, niter)
                chunk_result = batched_test(start, end) # type: ignore
                res = res.write(i, chunk_result)
                return i+1, res

            _, res = tf.while_loop(lambda i, res: i < nchunks, body, [0, res])

            return res.stack()

        start_calculation()
        
        reset_random_seeds(seed = seed)
        
        frob_norm = compute_test(max_vectorize = max_vectorize) # type: ignore
                             
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "FN Test_tf"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "tensorflow"}}
        result_value: Dict[str, Any] = {"metric_list": frob_norm.numpy()}
        result = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)