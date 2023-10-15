__all__ = ["LRMetric"]

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


def lr_statistic_np(logprob_ref_ref: DataTypeNP,
                    logprob_ref_alt: DataTypeNP,
                    logprob_alt_alt: DataTypeNP,
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Compute number of samples
    n_ref = len(logprob_ref_ref)
    n_alt = len(logprob_ref_alt)
    
    # Create masks for finite log probabilities
    finite_indices_ref_ref = np.isfinite(logprob_ref_ref)
    finite_indices_ref_alt = np.isfinite(logprob_ref_alt)
    finite_indices_alt_alt = np.isfinite(logprob_alt_alt)
    
    # Count the number of finite samples
    n_ref_finite = np.sum(finite_indices_ref_ref.astype(np.int32))
    n_alt_finite = np.sum((finite_indices_alt_alt & finite_indices_ref_alt).astype(np.int32))

    if n_ref_finite < n_ref:
        fraction = (n_ref - n_ref_finite) / n_ref
        print(f"Warning: Removed a fraction {fraction} of reference samples due to non-finite log probabilities.")
        
    if n_alt_finite < n_alt:
        fraction = (n_alt - n_alt_finite) / n_alt
        print(f"Warning: Removed a fraction {fraction} of alternative samples due to non-finite log probabilities.")
    
    # Combined finite indices
    combined_finite_indices = finite_indices_ref_ref & finite_indices_ref_alt & finite_indices_alt_alt
    
    # Use masks to filter the reshaped log probabilities
    logprob_ref_ref_filtered = np.where(combined_finite_indices, logprob_ref_ref, 0.)
    logprob_ref_alt_filtered = np.where(combined_finite_indices, logprob_ref_alt, 0.)
    logprob_alt_alt_filtered = np.where(combined_finite_indices, logprob_alt_alt, 0.)
    
    # Compute log likelihoods
    logprob_ref_ref_sum = np.sum(logprob_ref_ref_filtered)
    logprob_ref_alt_sum = np.sum(logprob_ref_alt_filtered)
    logprob_alt_alt_sum = np.sum(logprob_alt_alt_filtered)
    lik_ref_dist = logprob_ref_ref_sum + logprob_ref_alt_sum
    lik_alt_dist = logprob_ref_ref_sum + logprob_alt_alt_sum
    
    # Compute likelihood ratio statistic
    lik_ratio = 2 * (lik_alt_dist - lik_ref_dist)
    
    # Casting to float32 before performing division
    n_ref_finite_float = float(n_ref_finite)
    n_alt_finite_float = float(n_alt_finite)

    # Compute normalized likelihood ratio statistic
    n = 2 * n_ref_finite_float * n_alt_finite_float / (n_ref_finite_float + n_alt_finite_float)
    
    # Compute normalized likelihood ratio statistic
    lik_ratio_norm = lik_ratio / np.sqrt(n)
    
    return logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm

@tf.function(experimental_compile=True)
def lr_statistic_tf(logprob_ref_ref: DataTypeTF,
                    logprob_ref_alt: DataTypeTF,
                    logprob_alt_alt: DataTypeTF
                   ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    # Compute number of samples
    n_ref = tf.shape(logprob_ref_ref)[0] # type: ignore
    n_alt = tf.shape(logprob_ref_alt)[0] # type: ignore
        
    # Create masks for finite log probabilities
    finite_indices_ref_ref = tf.math.is_finite(logprob_ref_ref)
    finite_indices_ref_alt = tf.math.is_finite(logprob_ref_alt)
    finite_indices_alt_alt = tf.math.is_finite(logprob_alt_alt)
    
    # Count the number of finite samples
    n_ref_finite = tf.reduce_sum(tf.cast(finite_indices_ref_ref, tf.int32))
    n_alt_finite = tf.reduce_sum(tf.cast(tf.math.logical_and(finite_indices_alt_alt, finite_indices_ref_alt), tf.int32))

    if n_ref_finite < n_ref:
        tf.print("Warning: Removed a fraction of reference samples due to non-finite log probabilities.")
        
    if n_alt_finite < n_alt:
        tf.print(f"Warning: Removed a fraction of alternative samples due to non-finite log probabilities.")
    
    # Combined finite indices
    combined_finite_indices = tf.math.logical_and(tf.math.logical_and(finite_indices_ref_ref, finite_indices_ref_alt), finite_indices_alt_alt)
    
    # Use masks to filter the reshaped log probabilities
    logprob_ref_ref_filtered = tf.where(combined_finite_indices, logprob_ref_ref, 0.)
    logprob_ref_alt_filtered = tf.where(combined_finite_indices, logprob_ref_alt, 0.)
    logprob_alt_alt_filtered = tf.where(combined_finite_indices, logprob_alt_alt, 0.)
    
    # Compute log likelihoods
    logprob_ref_ref_sum = tf.reduce_sum(logprob_ref_ref_filtered)
    logprob_ref_alt_sum = tf.reduce_sum(logprob_ref_alt_filtered)
    logprob_alt_alt_sum = tf.reduce_sum(logprob_alt_alt_filtered)
    lik_ref_dist = logprob_ref_ref_sum + logprob_ref_alt_sum
    lik_alt_dist = logprob_ref_ref_sum + logprob_alt_alt_sum
    
    # Compute likelihood ratio statistic
    lik_ratio = 2 * (lik_alt_dist - lik_ref_dist)
    
    # Casting to float32 before performing division
    n_ref_finite_float = tf.cast(n_ref_finite, tf.float32)
    n_alt_finite_float = tf.cast(n_alt_finite, tf.float32)  

    # Compute normalized likelihood ratio statistic
    n = 2 * n_ref_finite_float * n_alt_finite_float / (n_ref_finite_float + n_alt_finite_float) # type: ignore
    
    # Compute normalized likelihood ratio statistic
    lik_ratio_norm = lik_ratio / tf.sqrt(tf.cast(n, lik_ratio.dtype)) # type: ignore
    
    return logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm


class LRMetric(TwoSampleTestBase):
    """
    Class for computing the Likelihood Ratio (LR) metric.
    The metric can be computed only if the `is_symb_1` and `is_symb_2` attributes of the `data_input` object are True.
    In the opposite case the metric is not computed and the results are set to None.
    It inherits from the TwoSampleTestBase class.
    The LR is computed by first computing the log probabilities of the reference and alternative samples
    under the reference and alternative distributions, and then computing the LR statistic.
    The LR statistic can be computed using either numpy or tensorflow.
    
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
        Function that computes the LR metric selecting among the Test_np and Test_tf methods depending on the value of the use_tf attribute.
        
    Test_np() -> None
        Function that computes the LR metric using numpy functions.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
        
    Test_tf() -> None
        Function that computes the LR metric using tensorflow functions.
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
        data_input = GMetrics.TwoSampleTestInputs(dist_1_input = dist_1_symb,
                                                  dist_2_input = dist_2_symb,
                                                  niter = 100,
                                                  batch_size = 10000,
                                                  dtype_input = tf.float64,
                                                  seed_input = 0,
                                                  use_tf = True,
                                                  verbose = True)

        # Compute LR metric
        LR_metric = GMetrics.LRMetric(data_input = data_input, 
                                      progress_bar = True, 
                                      verbose = True)
        LR_metric.compute()
        LR_metric.Results[0].result_value
        
        >> {"logprob_ref_ref_sum_list": ...,
            "logprob_ref_alt_sum_list": ..., 
            "logprob_alt_alt_sum_list": ...,
            "lik_ratio_list": ...,
            "lik_ratio_norm_list": ...}
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
        
    def compute(self) -> None:
        """
        Function that computes the LR metric selecting among the Test_np and Test_tf 
        methods depending on the value of the use_tf attribute.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

        """
        if self.use_tf:
            self.Test_tf()
        else:
            self.Test_np()
    
    def Test_np(self) -> None:
        """
        Function that computes the LR metric using numpy functions.
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
        # Utility function
        def stop_calculation() -> None:
            timestamp: str = datetime.now().isoformat()
            test_name: str = "LR Test_np"
            parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy", "note": "test skipped because the inputs are not symbolic."}}
            result_value: Dict[str, Optional[DataTypeTF]] = {"logprob_ref_ref_sum_list": None,
                                                             "logprob_ref_alt_sum_list": None, 
                                                             "logprob_alt_alt_sum_list": None,
                                                             "lik_ratio_list": None,
                                                             "lik_ratio_norm_list": None} # type: ignore
            result = TwoSampleTestResult(timestamp, test_name, parameters, result_value) # type: ignore
            self.Results.append(result)
            raise ValueError("LR metric can be computed only if the inputs are symbolic tfd.Distribution objects. Metric result has been set to `None`.")
        
        # Set alias for inputs
        if isinstance(self.Inputs.dist_1_symb, tfp.distributions.Distribution):
            dist_1_symb: tfp.distributions.Distribution = self.Inputs.dist_1_symb
        else:
            stop_calculation()
        if isinstance(self.Inputs.dist_2_symb, tfp.distributions.Distribution):
            dist_2_symb: tfp.distributions.Distribution = self.Inputs.dist_2_symb
        else:
            stop_calculation()
        if not self.Inputs.is_symb_1 or not self.Inputs.is_symb_2:
            stop_calculation()
        if isinstance(self.Inputs.dist_1_num, np.ndarray):
            dist_1_num: DataTypeNP = self.Inputs.dist_1_num
        else:
            dist_1_num = self.Inputs.dist_1_num.numpy() # type: ignore
        if isinstance(self.Inputs.dist_2_num, np.ndarray):
            dist_2_num: DataTypeNP = self.Inputs.dist_2_num
        else:
            dist_2_num = self.Inputs.dist_2_num.numpy() # type: ignore
        ndims: int = self.Inputs.ndims
        niter: int
        batch_size: int
        niter, batch_size = self.get_niter_batch_size_np() # type: ignore
        if isinstance(self.Inputs.dtype, tf.DType):
            dtype: Union[type, np.dtype] = self.Inputs.dtype.as_numpy_dtype
        else:
            dtype = self.Inputs.dtype
        seed: int = self.Inputs.seed
        dist_1_k: tf.Tensor
        dist_2_k: tf.Tensor
        
        # Utility functions
        def start_calculation() -> None:
            conditional_print(self.verbose, "\n------------------------------------------")
            conditional_print(self.verbose, "Starting LR metric calculation...")
            conditional_print(self.verbose, "niter = {}" .format(niter))
            conditional_print(self.verbose, "batch_size = {}" .format(batch_size))
            self._start = timer()
            
        def init_progress_bar() -> None:
            nonlocal niter
            if self.progress_bar:
                self.pbar = tqdm(total = niter, desc = "Iterations")

        def update_progress_bar() -> None:
            if not self.pbar.disable:
                self.pbar.update(1)

        def close_progress_bar() -> None:
            if not self.pbar.disable:
                self.pbar.close()

        def end_calculation() -> None:
            self._end = timer()
            conditional_print(self.verbose, "Two-sample test calculation completed in "+str(self.end-self.start)+" seconds.")
                
        logprob_ref_ref_sum_list: List[DataTypeNP] = []
        logprob_ref_alt_sum_list: List[DataTypeNP] = []
        logprob_alt_alt_sum_list: List[DataTypeNP] = []
        lik_ratio_list: List[DataTypeNP] = []
        lik_ratio_norm_list: List[DataTypeNP] = []

        start_calculation()
        init_progress_bar()
            
        reset_random_seeds(seed = seed)
        
        conditional_print(self.verbose, "Running numpy LR calculation...")
        
        for k in range(niter):
            if not np.shape(dist_1_num[0])[0] == 0 and not np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = tf.convert_to_tensor(dist_1_num[k*batch_size:(k+1)*batch_size,:])
                dist_2_k = tf.convert_to_tensor(dist_2_num[k*batch_size:(k+1)*batch_size,:])
            elif not np.shape(dist_1_num[0])[0] == 0 and np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = tf.convert_to_tensor(dist_1_num[k*batch_size:(k+1)*batch_size,:])
                dist_2_k = tf.cast(dist_2_symb.sample(batch_size), dtype = dtype) # type: ignore
            elif np.shape(dist_1_num[0])[0] == 0 and not np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = tf.cast(dist_1_symb.sample(batch_size), dtype = dtype) # type: ignore
                dist_2_k = tf.convert_to_tensor(dist_2_num[k*batch_size:(k+1)*batch_size,:])
            else:
                dist_1_k = tf.cast(dist_1_symb.sample(batch_size), dtype = dtype) # type: ignore
                dist_2_k = tf.cast(dist_2_symb.sample(batch_size), dtype = dtype) # type: ignore
            logprob_ref_ref = dist_1_symb.log_prob(dist_1_k).numpy() # type: ignore
            logprob_ref_alt = dist_1_symb.log_prob(dist_2_k).numpy() # type: ignore
            logprob_alt_alt = dist_2_symb.log_prob(dist_2_k).numpy() # type: ignore
            logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm = lr_statistic_np(logprob_ref_ref, 
                                                                                                                       logprob_ref_alt, 
                                                                                                                       logprob_alt_alt)
            logprob_ref_ref_sum_list.append(logprob_ref_ref_sum)
            logprob_ref_alt_sum_list.append(logprob_ref_alt_sum)
            logprob_alt_alt_sum_list.append(logprob_alt_alt_sum)
            lik_ratio_list.append(lik_ratio)
            lik_ratio_norm_list.append(lik_ratio_norm)
            update_progress_bar()
        
        close_progress_bar()
        end_calculation()
        
        timestamp = datetime.now().isoformat()
        test_name = "LR Test_np"
        parameters = {**self.param_dict, **{"backend": "numpy"}}
        result_value = {"logprob_ref_ref_sum_list": np.array(logprob_ref_ref_sum_list),
                        "logprob_ref_alt_sum_list": np.array(logprob_ref_alt_sum_list),
                        "logprob_alt_alt_sum_list": np.array(logprob_alt_alt_sum_list),
                        "lik_ratio_list": np.array(lik_ratio_list),
                        "lik_ratio_norm_list": np.array(lik_ratio_norm_list)} # type: ignore
        result = TwoSampleTestResult(timestamp, test_name, parameters, result_value) # type: ignore
        self.Results.append(result)
        
    def Test_tf(self) -> None:
        """
        Function that computes the Frobenuis norm between the correlation matrices 
        of the two samples using tensorflow functions.
        The number of random directions used for the projection is given by nslices.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
        
        Parameters:
        -----------
        nslices: int, optional, default = 100
            Number of random directions to use for the projection.

        Returns:
        --------
        None
        """
        # Utility function
        def stop_calculation() -> None:
            timestamp: str = datetime.now().isoformat()
            test_name: str = "LR Test_np"
            parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy", "note": "test skipped because the inputs are not symbolic."}}
            result_value: Dict[str, Optional[DataTypeTF]] = {"logprob_ref_ref_sum_list": None,
                                                             "logprob_ref_alt_sum_list": None, 
                                                             "logprob_alt_alt_sum_list": None,
                                                             "lik_ratio_list": None,
                                                             "lik_ratio_norm_list": None} # type: ignore
            result = TwoSampleTestResult(timestamp, test_name, parameters, result_value) # type: ignore
            self.Results.append(result)
            raise ValueError("LR metric can be computed only if the inputs are symbolic tfd.Distribution objects. Metric result has been set to `None`.")
        
        # Set alias for inputs
        if isinstance(self.Inputs.dist_1_symb, tfp.distributions.Distribution):
            dist_1_symb: tfp.distributions.Distribution = self.Inputs.dist_1_symb
        else:
            stop_calculation()
        if isinstance(self.Inputs.dist_2_symb, tfp.distributions.Distribution):
            dist_2_symb: tfp.distributions.Distribution = self.Inputs.dist_2_symb
        else:
            stop_calculation()
        if not self.Inputs.is_symb_1 or not self.Inputs.is_symb_2:
            stop_calculation()
        if isinstance(self.Inputs.dist_1_num, np.ndarray):
            dist_1_num: tf.Tensor = tf.convert_to_tensor(self.Inputs.dist_1_num)
        else:
            dist_1_num = self.Inputs.dist_1_num # type: ignore
        if isinstance(self.Inputs.dist_2_num, np.ndarray):
            dist_2_num: tf.Tensor = tf.convert_to_tensor(self.Inputs.dist_2_num)
        else:
            dist_2_num = self.Inputs.dist_2_num # type: ignore
        ndims: int = self.Inputs.ndims
        niter: int
        batch_size: int
        niter, batch_size = [int(i) for i in self.get_niter_batch_size_tf()] # type: ignore
        dtype: tf.DType = tf.as_dtype(self.Inputs.dtype)
        seed: int = self.Inputs.seed
        
        # Utility functions
        def start_calculation() -> None:
            conditional_tf_print(self.verbose, "\n------------------------------------------")
            conditional_tf_print(self.verbose, "Starting LR metric calculation...")
            conditional_tf_print(self.verbose, "Running TF LR calculation...")
            conditional_tf_print(self.verbose, "niter =", niter)
            conditional_tf_print(self.verbose, "batch_size =", batch_size)
            self._start = timer()

        def end_calculation() -> None:
            self._end = timer()
            elapsed = self.end - self.start
            conditional_tf_print(self.verbose, "LR metric calculation completed in", str(elapsed), "seconds.")
            
        def set_dist_num_from_symb(dist: DistTypeTF,
                                   nsamples: int,
                                   seed: int = 0
                                  ) -> tf.Tensor:
            nonlocal dtype
            #dist_num: tf.Tensor = tf.cast(dist.sample(nsamples, seed = int(seed)), dtype = dtype) # type: ignore
            dist_num: tf.Tensor = generate_and_clean_data(dist, nsamples, 1000, dtype = self.Inputs.dtype, seed = int(seed), mirror_strategy = self.Inputs.mirror_strategy) # type: ignore
            return dist_num
        
        def return_dist_num(dist_num: tf.Tensor) -> tf.Tensor:
            return dist_num
        
        #@tf.function(reduce_retracing=True)
        def compute_test() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            # Check if numerical distributions are empty and print a warning if so
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_1_num tensor is empty. Batches will be generated 'on-the-fly' from dist_1_symb.") # type: ignore
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_2_num tensor is empty. Batches will be generated 'on-the-fly' from dist_2_symb.") # type: ignore
            
            # Initialize the result TensorArray
            res = tf.TensorArray(dtype, size = niter)
            
            # Define unique constants for the two distributions. It is sufficient that these two are different to get different samples from the two distributions, if they are equal. 
            # There is not problem with subsequent calls to the batched_test function, since the random state is updated at each call.
            seed_dist_1  = int(1e6)  # Seed for distribution 1
            seed_dist_2  = int(1e12)  # Seed for distribution 2
    
            def body(i, res):
                
                # Define the loop body to vectorize over ndims*chunk_size
                dist_1_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_1_symb, nsamples = batch_size, seed = seed_dist_1), # type: ignore
                                               false_fn = lambda: return_dist_num(dist_1_num[i * batch_size: (i + 1) * batch_size, :])) # type: ignore
                dist_2_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_2_symb, nsamples = batch_size, seed = seed_dist_2), # type: ignore
                                               false_fn = lambda: return_dist_num(dist_2_num[i * batch_size: (i + 1) * batch_size, :])) # type: ignore
                logprob_ref_ref = dist_1_symb.log_prob(dist_1_k) # type: ignore
                logprob_ref_alt = dist_1_symb.log_prob(dist_2_k) # type: ignore
                logprob_alt_alt = dist_2_symb.log_prob(dist_2_k) # type: ignore
                logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm = lr_statistic_tf(logprob_ref_ref, 
                                                                                                                           logprob_ref_alt, 
                                                                                                                           logprob_alt_alt) # type: ignore
                logprob_ref_ref_sum = tf.cast(logprob_ref_ref_sum, dtype=dtype)
                logprob_ref_alt_sum = tf.cast(logprob_ref_alt_sum, dtype=dtype)
                logprob_alt_alt_sum = tf.cast(logprob_alt_alt_sum, dtype=dtype)
                lik_ratio = tf.cast(lik_ratio, dtype=dtype)
                lik_ratio_norm = tf.cast(lik_ratio_norm, dtype=dtype)
                
                result_value = tf.stack([logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm]) # type: ignore
                
                res = res.write(i, result_value)
                return i+1, res
    
            _, res = tf.while_loop(lambda i, _: i < niter, body, [0, res])
        
            return tf.transpose(res.stack()) # type: ignore

        start_calculation()
        
        reset_random_seeds(seed = seed)
        
        logprob_ref_ref_sum_list, logprob_ref_alt_sum_list, logprob_alt_alt_sum_list, lik_ratio_list, lik_ratio_norm_list = compute_test() # type: ignore
                             
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "LR Test_tf"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "tensorflow"}}
        result_value = {"logprob_ref_ref_sum_list": logprob_ref_ref_sum_list.numpy(),
                        "logprob_ref_alt_sum_list": logprob_ref_alt_sum_list.numpy(),
                        "logprob_alt_alt_sum_list": logprob_alt_alt_sum_list.numpy(),
                        "lik_ratio_list": lik_ratio_list.numpy(),
                        "lik_ratio_norm_list": lik_ratio_norm_list.numpy()} # type: ignore
        result = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)