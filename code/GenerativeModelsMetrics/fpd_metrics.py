__all__ = ["_linear",
           "_matrix_sqrtm",
           "_calculate_frechet_distance_tf",
           "_normalise_features_tf",
           "fpd_tf",
           "fpd_tf_fit",   
           "FPDMetric"]

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import traceback
from datetime import datetime
from timeit import default_timer as timer
from tqdm import tqdm # type: ignore
from scipy.stats import ks_2samp # type: ignore
from scipy.optimize import curve_fit # type: ignore
from .utils import reset_random_seeds
from .utils import conditional_print
from .utils import conditional_tf_print
from .utils import generate_and_clean_data
from .utils import NumpyDistribution
from .base import TwoSampleTestInputs
from .base import TwoSampleTestBase
from .base import TwoSampleTestResult
from .base import TwoSampleTestResults
from jetnet.evaluation import gen_metrics as JMetrics # type: ignore

from typing import Tuple, Union, Optional, Type, Dict, Any, List, Set
from .utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType

def _linear(x, intercept, slope):
    return intercept + slope * x

@tf.function(jit_compile=True, reduce_retracing=True)
def _matrix_sqrtm(matrix: tf.Tensor) -> tf.Tensor:
    # Eigenvalue decomposition
    e: tf.Tensor
    v: tf.Tensor
    e, v = tf.linalg.eigh(matrix)
    # Create the square root of the diagonal matrix with the eigenvalues
    sqrt_e: tf.Tensor
    sqrt_e = tf.sqrt(e)
    sqrt_e = tf.linalg.diag(sqrt_e)
    # Reconstruct the square root of the original matrix
    sqrt_mat: tf.Tensor
    sqrt_mat = tf.linalg.matmul(tf.linalg.matmul(v, sqrt_e), tf.linalg.adjoint(v))
    return sqrt_mat

@tf.function(jit_compile=True, reduce_retracing=True)
def _calculate_frechet_distance_tf(mu1_input: DataType,
                                   sigma1_input: DataType,
                                   mu2_input: DataType,
                                   sigma2_input: DataType,
                                   eps: float = 1e-6
                                  ) -> tf.Tensor:
    """
    TensorFlow implementation of the Frechet Distance.
    """
    mu1: tf.Tensor = tf.expand_dims(mu1_input, axis=-1) if len(tf.shape(mu1_input)) == 1 else tf.convert_to_tensor(mu1_input)
    mu2: tf.Tensor = tf.expand_dims(mu2_input, axis=-1) if len(tf.shape(mu1_input)) == 1 else tf.convert_to_tensor(mu2_input)
    sigma1 = tf.convert_to_tensor(sigma1_input, dtype=tf.float32)
    sigma2 = tf.convert_to_tensor(sigma2_input, dtype=tf.float32)

    diff: tf.Tensor = mu1 - mu2

    # Product might be almost singular
    covmean_sqrtm: tf.Tensor = _matrix_sqrtm(tf.linalg.matmul(sigma1, sigma2)) # type: ignore
    
    # Handle possible numerical errors
    if not tf.reduce_all(tf.math.is_finite(covmean_sqrtm)):
        offset: tf.Tensor = tf.cast(tf.eye(tf.shape(sigma1)[0]) * eps, sigma1.dtype) # type: ignore
        
        covmean_sqrtm: tf.Tensor = _matrix_sqrtm(tf.linalg.matmul(sigma1 + offset, sigma2 + offset)) # type: ignore

    # Handle possible imaginary component
    if tf.reduce_any(tf.math.imag(covmean_sqrtm) != 0):
        covmean_sqrtm = tf.math.real(covmean_sqrtm)

    tr_covmean: tf.Tensor = tf.linalg.trace(covmean_sqrtm)

    frechet_distance: tf.Tensor = tf.reduce_sum(diff * diff) + tf.linalg.trace(sigma1) + tf.linalg.trace(sigma2) - 2.0 * tr_covmean

    return frechet_distance

@tf.function(jit_compile = True, reduce_retracing = True)
def _normalise_features_tf(data1_input: DataType, 
                           data2_input: Optional[DataType] = None
                          ) -> Union[DataTypeTF, Tuple[DataTypeTF, DataTypeTF]]:
    data1: DataTypeTF = tf.convert_to_tensor(data1_input)
    maxes: tf.Tensor = tf.reduce_max(tf.abs(data1), axis=0)
    maxes = tf.where(tf.equal(maxes, 0), tf.ones_like(maxes), maxes)  # don't normalize in case of features which are just 0

    if data2_input is not None:
        data2: DataTypeTF = tf.convert_to_tensor(data2_input)
        return data1 / maxes, data2 / maxes
    else:
        return data1 / maxes
    
@tf.function(jit_compile = True, reduce_retracing = True)
def fpd_tf(data1_input: DataType, 
           data2_input: DataType,
           min_samples: int = 20_000, 
           max_samples: int = 50_000, 
           num_batches: int = 20, 
           num_points: int = 10,
           normalise: bool = True,
           seed: int = 0
          ) -> Tuple[DataTypeTF, tf.Tensor]:
    data1: DataTypeTF
    data2: DataTypeTF
    if normalise:
        data1, data2 = _normalise_features_tf(data1_input, data2_input) # type: ignore
    else:
        data1 = tf.convert_to_tensor(data1_input)
        data2 = tf.convert_to_tensor(data2_input)

    if len(data1.shape) == 1: # type: ignore
        data1 = tf.expand_dims(data1, axis=-1)
    if len(data2.shape) == 1: # type: ignore
        data2 = tf.expand_dims(data2, axis=-1)
        
    # Preallocate random indices
    max_batch_size: int = max_samples
    total_batches: int = num_points * num_batches
    all_rand1: tf.Tensor = tf.random.uniform(shape=[total_batches, max_batch_size], minval=0, maxval=tf.shape(data1)[0], dtype=tf.int32, seed=seed)
    all_rand2: tf.Tensor = tf.random.uniform(shape=[total_batches, max_batch_size], minval=0, maxval=tf.shape(data2)[0], dtype=tf.int32, seed=seed)
    batches: tf.Tensor = tf.cast(1 / tf.linspace(1.0 / min_samples, 1.0 / max_samples, num_points), dtype=tf.int32) # type: ignore
    
    vals: tf.TensorArray = tf.TensorArray(dtype = data1.dtype, 
                                          size = num_points)
    
    counter: int = 0
    for i in tf.range(tf.shape(batches)[0]):
        batch_size: int = batches[i]
        val_points: tf.TensorArray = tf.TensorArray(dtype = data1.dtype,
                                                    size = num_batches)

        for j in tf.range(num_batches):
            rand1: tf.Tensor = all_rand1[counter, :batch_size]
            rand2: tf.Tensor = all_rand2[counter, :batch_size]
            counter += 1

            rand_sample1: tf.Tensor = tf.gather(data1, rand1)
            rand_sample2: tf.Tensor = tf.gather(data2, rand2)

            mu1: tf.Tensor = tf.reduce_mean(rand_sample1, axis=0)
            mu2: tf.Tensor = tf.reduce_mean(rand_sample2, axis=0)
            sigma1: tf.Tensor = tfp.stats.covariance(rand_sample1, sample_axis=0, event_axis=-1)
            sigma2: tf.Tensor = tfp.stats.covariance(rand_sample2, sample_axis=0, event_axis=-1)

            val: tf.Tensor = _calculate_frechet_distance_tf(mu1, sigma1, mu2, sigma2) # type: ignore
            val_points = val_points.write(j, val)

        val_points = val_points.stack()
        vals = vals.write(i, tf.reduce_mean(val_points))

    vals_stacked: DataTypeTF = vals.stack()
    
    return vals_stacked, batches

def fpd_tf_fit(vals_list_input: DataType,
               batches_list_input: tf.Tensor
              ) -> Tuple[DataTypeNP, DataTypeNP]:
    vals_list: DataTypeNP = np.array(vals_list_input)
    batches_list: DataTypeNP = np.array(batches_list_input)
    metric_list: list = []
    metric_error_list: list = []
    for vals, batches in zip(vals_list, batches_list):
        params: DataTypeNP
        covs: DataTypeNP
        params, covs = curve_fit(_linear, 1 / batches, vals, bounds=([0, 0], [np.inf, np.inf]))
        metric_list.append(params[0])
        metric_error_list.append(np.sqrt(np.diag(covs)[0]))
    return np.array(metric_list), np.array(metric_error_list)

class FPDMetric(TwoSampleTestBase):
    """
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 progress_bar: bool = False,
                 verbose: bool = False,
                 **fpd_kwargs
                ) -> None:
        # From base class
        self._Inputs: TwoSampleTestInputs
        self._progress_bar: bool
        self._verbose: bool
        self._start: float
        self._end: float
        self._pbar: tqdm
        self._Results: TwoSampleTestResults
        
        # New attributes
        self.fpd_kwargs = fpd_kwargs # Use the setter to validate arguments
        
        super().__init__(data_input = data_input, 
                         progress_bar = progress_bar,
                         verbose = verbose)
        
    @property
    def fpd_kwargs(self) -> Dict[str, Any]:
        return self._fpd_kwargs
    
    @fpd_kwargs.setter
    def fpd_kwargs(self, fpd_kwargs: Dict[str, Any]) -> None:
        valid_keys: Set[str] = {'min_samples', 'max_samples', 'num_batches', 'num_points', 'normalise', 'seed'}
        # Dynamically get valid keys from `fpd` function's parameters
        # valid_keys = set(inspect.signature(JMetrics.fpd).parameters.keys())
        
        for key in fpd_kwargs.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid key: {key}. Valid keys are {valid_keys}")

        # You can add more specific validations for each argument here
        if 'min_samples' in fpd_kwargs and (not isinstance(fpd_kwargs['min_samples'], int) or fpd_kwargs['min_samples'] <= 0):
            raise ValueError("min_samples must be a positive integer")
            
        if 'max_samples' in fpd_kwargs and (not isinstance(fpd_kwargs['max_samples'], int) or fpd_kwargs['max_samples'] <= 0):
            raise ValueError("max_samples must be a positive integer")
        
        if 'num_batches' in fpd_kwargs and (not isinstance(fpd_kwargs['num_batches'], int) or fpd_kwargs['num_batches'] <= 0):
            raise ValueError("num_batches must be a positive integer")
            
        if 'num_points' in fpd_kwargs and (not isinstance(fpd_kwargs['num_points'], int) or fpd_kwargs['num_points'] <= 0):
            raise ValueError("num_points must be a positive integer")
        
        if 'normalise' in fpd_kwargs and not isinstance(fpd_kwargs['normalise'], bool):
            raise ValueError("normalise must be a boolean")
        
        if 'seed' in fpd_kwargs and not isinstance(fpd_kwargs['seed'], int):
            raise ValueError("seed must be an integer")
        
        self._fpd_kwargs = fpd_kwargs
        
            
    def compute(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the FPD  metric (and its uncertainty) from two multivariate samples
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
    
    def Test_np(self, **fpd_kwargs) -> None:
        """
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
        dist_1_k: DataTypeNP
        dist_2_k: DataTypeNP
        
        # Utility functions
        def set_dist_num_from_symb(dist: DistType,
                                   nsamples: int,
                                   dtype: Union[type, np.dtype],
                                  ) -> DataTypeNP:
            if isinstance(dist, tfp.distributions.Distribution):
                dist_num_tmp: DataTypeTF = generate_and_clean_data(dist, nsamples, self.Inputs.batch_size_gen, dtype = dtype, seed_generator = self.Inputs.seed_generator, mirror_strategy = self.Inputs.mirror_strategy) # type: ignore
                dist_num: DataTypeNP = dist_num_tmp.numpy().astype(dtype) # type: ignore
            elif isinstance(dist, NumpyDistribution):
                dist_num = dist.sample(nsamples).astype(dtype = dtype)
            else:
                raise TypeError("dist must be either a tfp.distributions.Distribution or a NumpyDistribution object.")
            return dist_num
        
        def start_calculation() -> None:
            conditional_print(self.verbose, "\n------------------------------------------")
            conditional_print(self.verbose, "Starting FPD metric calculation...")
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
        
        metric_list: List[float] = []
        metric_error_list: List[float] = []

        start_calculation()
        init_progress_bar()
            
        self.Inputs.reset_seed_generator()
        
        conditional_print(self.verbose, "Running numpy FPD calculation...")
        for k in range(niter):
            if not np.shape(dist_1_num[0])[0] == 0 and not np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            elif not np.shape(dist_1_num[0])[0] == 0 and np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = dist_1_num[k*batch_size:(k+1)*batch_size,:]
                dist_2_k = set_dist_num_from_symb(dist = dist_2_symb, nsamples = batch_size, dtype = dtype)
            elif np.shape(dist_1_num[0])[0] == 0 and not np.shape(dist_2_num[0])[0] == 0:
                dist_1_k = set_dist_num_from_symb(dist = dist_1_symb, nsamples = batch_size, dtype = dtype)
                dist_2_k = dist_2_num[k*batch_size:(k+1)*batch_size,:]
            else:
                dist_1_k = set_dist_num_from_symb(dist = dist_1_symb, nsamples = batch_size, dtype = dtype)
                dist_2_k = set_dist_num_from_symb(dist = dist_2_symb, nsamples = batch_size, dtype = dtype)
            metric: float
            metric_error: float
            metric, metric_error = JMetrics.fpd(dist_1_k, dist_2_k, **fpd_kwargs)
            metric_list.append(metric)
            metric_error_list.append(metric_error)
            update_progress_bar()
        
        close_progress_bar()
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "FPD Test_np"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy"}}
        result_value: Dict[str, Optional[DataTypeNP]] = {"metric_list": np.array(metric_list),
                                                         "metric_error_list": np.array(metric_error_list)}
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)
        
    def Test_tf(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the FPD  metric (and its uncertainty) from two multivariate samples
        using tensorflow functions.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The calculation is parallelized over max_vectorize (out of niter).
        The results are stored in the Results attribute.

        Parameters:
        ----------
        max_vectorize: int, optional, default = 100
            A maximum number of batch_size*max_vectorize samples per time are processed by the tensorflow backend.
            Given a value of max_vectorize, the niter FPD calculations are split in chunks of max_vectorize.
            Each chunk is processed by the tensorflow backend in parallel. If ndims is larger than max_vectorize,
            the calculation is vectorized niter times over ndims.

        Returns:
        --------
        None
        """
        max_vectorize = int(max_vectorize)
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
            raise TypeError("dist_1_symb must be a tfp.distributions.Distribution object when use_tf is True.")
        if isinstance(self.Inputs.dist_2_symb, tfp.distributions.Distribution):
            dist_2_symb: tfp.distributions.Distribution = self.Inputs.dist_2_symb
        else:
            raise TypeError("dist_2_symb must be a tfp.distributions.Distribution object when use_tf is True.")
        ndims: int = self.Inputs.ndims
        niter: int
        batch_size: int
        niter, batch_size = [int(i) for i in self.get_niter_batch_size_tf()] # type: ignore
        dtype: tf.DType = tf.as_dtype(self.Inputs.dtype)
        
        # Utility functions
        def start_calculation() -> None:
            conditional_tf_print(self.verbose, "\n------------------------------------------")
            conditional_tf_print(self.verbose, "Starting FPD metric calculation...")
            conditional_tf_print(self.verbose, "Running TF FPD calculation...")
            conditional_tf_print(self.verbose, "niter =", niter)
            conditional_tf_print(self.verbose, "batch_size =", batch_size)
            self._start = timer()

        def end_calculation() -> None:
            self._end = timer()
            elapsed = self.end - self.start
            conditional_tf_print(self.verbose, "FPD metric calculation completed in", str(elapsed), "seconds.")
                    
        def set_dist_num_from_symb(dist: DistTypeTF,
                                   nsamples: int,
                                  ) -> tf.Tensor:
            dist_num: tf.Tensor = generate_and_clean_data(dist, nsamples, self.Inputs.batch_size_gen, dtype = self.Inputs.dtype, seed_generator = self.Inputs.seed_generator, mirror_strategy = self.Inputs.mirror_strategy) # type: ignore
            return dist_num
        
        def return_dist_num(dist_num: tf.Tensor) -> tf.Tensor:
            return dist_num
            
        def batched_test(start: tf.Tensor, 
                         end: tf.Tensor
                        ) -> DataTypeTF:
            # Define batched distributions
            dist_1_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_1_symb, nsamples = batch_size*(end-start)),
                                               false_fn = lambda: return_dist_num(dist_1_num[start*batch_size:end*batch_size, :])) # type: ignore
            dist_2_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_2_symb, nsamples = batch_size*(end-start)),
                                               false_fn = lambda: return_dist_num(dist_2_num[start*batch_size:end*batch_size, :])) # type: ignore

            dist_1_k = tf.reshape(dist_1_k, (end-start, batch_size, ndims))
            dist_2_k = tf.reshape(dist_2_k, (end-start, batch_size, ndims))

            # Define the loop body to vectorize over ndims*chunk_size
            def loop_body(idx):
                vals, batches = fpd_tf(dist_1_k[idx, :, :], dist_2_k[idx, :, :], **self.fpd_kwargs) # type: ignore
                vals = tf.cast(vals, dtype=dtype)
                batches = tf.cast(batches, dtype=dtype)
                return vals, batches

            # Vectorize over ndims*chunk_size
            vals_list: tf.Tensor
            batches_list: tf.Tensor
            vals_list, batches_list = tf.vectorized_map(loop_body, tf.range(end-start)) # type: ignore
            
            res: DataTypeTF = tf.concat([vals_list, batches_list], axis=1) # type: ignore
            #tf.print(f"vals shape: {vals_list.shape}")
            #tf.print(f"batches shape: {batches_list.shape}")
            #tf.print(f"res shape: {res.shape}")
    
            return res
        
        def compute_test(max_vectorize: int = 100) -> Tuple[DataTypeTF, tf.Tensor]:
            # Check if numerical distributions are empty and print a warning if so
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_1_num tensor is empty. Batches will be generated 'on-the-fly' from dist_1_symb.") # type: ignore
            conditional_tf_print(tf.logical_and(tf.equal(tf.shape(dist_1_num[0])[0],0),self.verbose), "The dist_2_num tensor is empty. Batches will be generated 'on-the-fly' from dist_2_symb.") # type: ignore
            
            # Ensure that max_vectorize is an integer larger than ndims
            max_vectorize = int(tf.cast(tf.minimum(max_vectorize, niter),tf.int32)) # type: ignore

            # Compute the maximum number of iterations per chunk
            max_iter_per_chunk: int = max_vectorize # type: ignore
            
            # Compute the number of chunks
            nchunks: int = int(tf.cast(tf.math.ceil(niter / max_iter_per_chunk), tf.int32)) # type: ignore
            conditional_tf_print(tf.logical_and(self.verbose,tf.logical_not(tf.equal(nchunks,1))), "nchunks =", nchunks) # type: ignore

            res: tf.TensorArray = tf.TensorArray(dtype, size = nchunks)
            res_vals: tf.TensorArray = tf.TensorArray(dtype, size = nchunks)
            res_batches: tf.TensorArray = tf.TensorArray(dtype, size = nchunks)

            def body(i: int, 
                     res: tf.TensorArray
                    ) -> Tuple[int, tf.TensorArray]:
                start: tf.Tensor = tf.cast(i * max_iter_per_chunk, tf.int32) # type: ignore
                end: tf.Tensor = tf.cast(tf.minimum(start + max_iter_per_chunk, niter), tf.int32) # type: ignore
                conditional_tf_print(tf.logical_and(tf.logical_or(tf.math.logical_not(tf.equal(start,0)),tf.math.logical_not(tf.equal(end,niter))), self.verbose), "Iterating from", start, "to", end, "out of", niter, ".") # type: ignore
                chunk_result: DataTypeTF = batched_test(start, end) # type: ignore
                res = res.write(i, chunk_result)
                return i+1, res
            
            def cond(i: int, 
                     res: tf.TensorArray):
                return i < nchunks
            
            _, res = tf.while_loop(cond, body, [0, res])
            
            for i in range(nchunks):
                res_i: DataTypeTF = tf.convert_to_tensor(res.read(i))
                npoints: tf.Tensor = res_i.shape[1] // 2 # type: ignore
                res_vals = res_vals.write(i, res_i[:, :npoints]) # type: ignore
                res_batches = res_batches.write(i, res_i[:, npoints:]) # type: ignore
                
            vals_list: DataTypeTF = tf.squeeze(res_vals.stack())
            batches_list: tf.Tensor = tf.squeeze(res_batches.stack())

            return vals_list, batches_list

        start_calculation()
        
        self.Inputs.reset_seed_generator()
        
        vals_list: DataTypeTF
        batches_list: tf.Tensor
        vals_list, batches_list  = compute_test(max_vectorize = max_vectorize)
                
        #print(f"vals_list: {vals_list=}")
        #print(f"batches_list: {batches_list=}")
        
        metric_list: DataTypeNP
        metric_error_list: DataTypeNP
        metric_list, metric_error_list = fpd_tf_fit(vals_list, batches_list)
                             
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "FPD Test_tf"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "tensorflow"}}
        result_value: Dict[str, Optional[DataTypeNP]] = {"metric_list": metric_list,
                                                         "metric_error_list": metric_error_list}
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)