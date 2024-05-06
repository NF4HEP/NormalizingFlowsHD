__all__ = ["_poly_kernel_pairwise_tf",
           "_mmd_quadratic_unbiased_tf",
           "_mmd_poly_quadratic_unbiased_tf",
           "_kpd_batches_tf",
           "kpd_tf",
           "KPDMetric"]

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import traceback
from datetime import datetime
from timeit import default_timer as timer
from tqdm import tqdm # type: ignore
from scipy.stats import iqr # type: ignore
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

@tf.function(jit_compile=True, reduce_retracing = True)
def _poly_kernel_pairwise_tf(X, Y, degree):
    gamma = tf.cast(1.0, X.dtype) / tf.cast(tf.shape(X)[-1], X.dtype)
    return tf.pow(tf.linalg.matmul(X, Y, transpose_b=True) * gamma + 1.0, degree)

@tf.function(jit_compile=True, reduce_retracing = True)
def _mmd_quadratic_unbiased_tf(XX, YY, XY):
    m = tf.cast(tf.shape(XX)[0], XX.dtype)
    n = tf.cast(tf.shape(YY)[0], YY.dtype)
    return (tf.reduce_sum(XX) - tf.linalg.trace(XX)) / (m * (m - 1)) \
           + (tf.reduce_sum(YY) - tf.linalg.trace(YY)) / (n * (n - 1)) \
           - 2 * tf.reduce_mean(XY)
           
@tf.function(jit_compile=True, reduce_retracing = True)
def _mmd_poly_quadratic_unbiased_tf(X, Y, degree=4):
    XX = _poly_kernel_pairwise_tf(X, X, degree=degree)
    YY = _poly_kernel_pairwise_tf(Y, Y, degree=degree)
    XY = _poly_kernel_pairwise_tf(X, Y, degree=degree)
    return _mmd_quadratic_unbiased_tf(XX, YY, XY)

@tf.function(jit_compile=True, reduce_retracing = True)
    
def kpd_tf(X: tf.Tensor,
           Y: tf.Tensor,
           num_batches: int = 10,
           batch_size: int = 5000,
           normalise: bool = True,
           seed: int = 42):
    vals_point = []
    for i in range(num_batches):
        tf.random.set_seed(seed + i * 1000)
        rand1 = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(X), dtype=tf.int32)
        rand2 = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(Y), dtype=tf.int32)
        
        rand_sample1 = tf.gather(X, rand1)
        rand_sample2 = tf.gather(Y, rand2)

        val = _mmd_poly_quadratic_unbiased_tf(rand_sample1, rand_sample2)
        vals_point.append(val)
    vals_point = tf.stack(vals_point)
    return vals_point

def kpd_tf_output(vals_points_input: DataTypeTF) -> DataTypeTF:
    vals_points: DistTypeNP = np.array(vals_points_input)
    metric_list: list = []
    metric_error_list: list = []
    for vals_point in vals_points:
        metric_list.append(np.median(vals_point))
        metric_error_list.append(iqr(vals_point, rng=(16.275, 83.725)) / 2)
    return np.array(metric_list), np.array(metric_error_list)
        
    # Calculating median and IQR using TensorFlow
    return 
           

class KPDMetric(TwoSampleTestBase):
    """
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 progress_bar: bool = False,
                 verbose: bool = False,
                 **kpd_kwargs
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
        self.kpd_kwargs = kpd_kwargs # Use the setter to validate arguments
        
        super().__init__(data_input = data_input, 
                         progress_bar = progress_bar,
                         verbose = verbose)
        
    @property
    def kpd_kwargs(self) -> Dict[str, Any]:
        return self._kpd_kwargs
    
    @kpd_kwargs.setter
    def kpd_kwargs(self, kpd_kwargs: Dict[str, Any]) -> None:
        valid_keys: Set[str] = {'num_batches', 'batch_size', 'normalise', 'seed'}
        # Dynamically get valid keys from `kpd` function's parameters
        # valid_keys = set(inspect.signature(JMetrics.kpd).parameters.keys())
        
        for key in kpd_kwargs.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid key: {key}. Valid keys are {valid_keys}")

        if 'num_batches' in kpd_kwargs and not isinstance(kpd_kwargs['num_batches'], int):
            raise ValueError("num_batches must be an integer")
        
        if 'batch_size' in kpd_kwargs and not isinstance(kpd_kwargs['batch_size'], int):
            raise ValueError("batch_size must be an integer")
        
        if 'normalise' in kpd_kwargs and not isinstance(kpd_kwargs['normalise'], bool):
            raise ValueError("normalise must be a boolean")
        
        if 'seed' in kpd_kwargs and not isinstance(kpd_kwargs['seed'], int):
            raise ValueError("seed must be an integer")
        
        self._kpd_kwargs = kpd_kwargs
        
            
    def compute(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the KPD  metric (and its uncertainty) from two multivariate samples
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
    
    def Test_np(self, **kpd_kwargs) -> None:
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
            conditional_print(self.verbose, "Starting KPD metric calculation...")
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
        
        conditional_print(self.verbose, "Running numpy KPD calculation...")
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
            metric, metric_error = JMetrics.kpd(dist_1_k, dist_2_k, **kpd_kwargs)
            metric_list.append(metric)
            metric_error_list.append(metric_error)
            update_progress_bar()
        
        close_progress_bar()
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "KPD Test_np"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy"}}
        result_value: Dict[str, Optional[DataTypeNP]] = {"metric_list": np.array(metric_list),
                                                         "metric_error_list": np.array(metric_error_list)}
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)
        
    def Test_tf(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the KPD  metric (and its uncertainty) from two multivariate samples
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
            Given a value of max_vectorize, the niter KPD calculations are split in chunks of max_vectorize.
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
            conditional_tf_print(self.verbose, "Starting KPD metric calculation...")
            conditional_tf_print(self.verbose, "Running TF KPD calculation...")
            conditional_tf_print(self.verbose, "niter =", niter)
            conditional_tf_print(self.verbose, "batch_size =", batch_size)
            self._start = timer()

        def end_calculation() -> None:
            self._end = timer()
            elapsed = self.end - self.start
            conditional_tf_print(self.verbose, "KPD metric calculation completed in", str(elapsed), "seconds.")
                    
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
                vals = kpd_tf(dist_1_k[idx, :, :], dist_2_k[idx, :, :], **self.kpd_kwargs) # type: ignore
                vals = tf.cast(vals, dtype=dtype)
                return vals

            # Vectorize over ndims*chunk_size
            vals_list: tf.Tensor = tf.vectorized_map(loop_body, tf.range(end-start)) # type: ignore
            
            res: DataTypeTF = vals_list
            #tf.print(f"vals shape: {vals_list.shape}")
            #tf.print(f"batches shape: {batches_list.shape}")
            #tf.print(f"res shape: {res.shape}")
    
            return res
        
        def compute_test(max_vectorize: int = 100) -> Tuple[DataTypeTF]:
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
            
            res_stacked: DataTypeTF = tf.reshape(res.stack(), (niter,))

            return res_stacked

        start_calculation()
        
        self.Inputs.reset_seed_generator()
        
        vals_list: DataTypeTF
        batches_list: tf.Tensor
        vals_list = compute_test(max_vectorize = max_vectorize)
                
        #print(f"vals_list: {vals_list=}")
        #print(f"batches_list: {batches_list=}")
        
        metric_list: DataTypeNP
        metric_error_list: DataTypeNP
        metric_list, metric_error_list = kpd_tf_output(vals_list)
                             
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "KPD Test_tf"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "tensorflow"}}
        result_value: Dict[str, Optional[DataTypeNP]] = {"metric_list": metric_list,
                                                         "metric_error_list": metric_error_list}
        result: TwoSampleTestResult = TwoSampleTestResult(timestamp, test_name, parameters, result_value)
        self.Results.append(result)