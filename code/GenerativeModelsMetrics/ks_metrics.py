__all__ = ["ks_2samp_tf",
           "_ks_2samp_tf_internal",
           "KSTest"]

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import traceback
from datetime import datetime
from timeit import default_timer as timer
from tqdm import tqdm # type: ignore
from scipy.stats import ks_2samp # type: ignore
from .utils import reset_random_seeds
from .utils import conditional_print
from .utils import conditional_tf_print
from .utils import NumpyDistribution
from .base import TwoSampleTestInputs
from .base import TwoSampleTestBase
from .base import TwoSampleTestResult
from .base import TwoSampleTestResults

from typing import Tuple, Union, Optional, Type, Dict, Any, List
from .utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType


@tf.function(reduce_retracing = True)
def ks_2samp_tf(data1: tf.Tensor, 
                data2: tf.Tensor,
                alternative: str = 'two-sided',
                method: str = 'auto',
                precision: int = 100,
                verbose: bool = False
               ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    This function is an API for the internal function _ks_2samp_tf_internal and is needed to
    convert the alternative and method arguments from strings to integer codes.
    
    The tf.function decorator is used to speed up subsequent calls to this function and to avoid retracing.

    See the documentation of _ks_2samp_tf_internal for more information on the implementation.

    Parameters:
    ----------
    Same as _ks_2samp_tf_internal.

    Returns: 
    --------
    Same as _ks_2samp_tf_internal.
    """
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("Invalid alternative.")
    if method not in ['auto', 'exact', 'asymp']:
        raise ValueError("Invalid method.")
    alternative_dict: Dict[str,int] = {'two-sided': 0, 'less': 1, 'greater': 2}
    method_dict: Dict[str,int] = {'auto': 0, 'exact': 1, 'asymp': 2}
    
    # Convert string input to integer codes.
    alternative_int: int = alternative_dict.get(alternative, 0)
    method_int: int = method_dict.get(method, 0)
    d: tf.Tensor
    prob: tf.Tensor
    d_location: tf.Tensor
    d, prob, d_location, d_sign = _ks_2samp_tf_internal(data1 = data1, 
                                                        data2 = data2, 
                                                        alternative_int = alternative_int, 
                                                        method_int = method_int,
                                                        precision = precision,
                                                        verbose = verbose) # type: ignore
    return d, prob, d_location, d_sign
    
    
@tf.function(reduce_retracing = True)
def _ks_2samp_tf_internal(data1: tf.Tensor, 
                          data2: tf.Tensor,
                          alternative_int: int = 0,
                          method_int: int = 0,
                          precision: int = 100,
                          verbose: bool = False
                         ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Internal function for computing the Kolmogorov-Smirnov test-statistic and p-value for two samples.
    See ks_2samp_tf for more information.
    
    Parameters:
    ----------
    data1: tf.Tensor, optional, shape = (n1,)
        First sample. Sample sizes can be different.
        
    data2: tf.Tensor, optional, shape = (n2,)
        Second sample. Sample sizes can be different.
        
    alternative: str, optional, default = 'two-sided'
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):
        - 'two-sided'
        - 'less': one-sided, see explanation in scipy.stats.ks_2samp function
        - 'greater': one-sided, see explanation in scipy.stats.ks_2samp function

    method: str, optional, default = 'auto'
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):
        - 'auto': use 'exact' for small size arrays, 'asymp' for larger
        - 'exact': use exact distribution of test statistic
        - 'asymp': use asymptotic distribution of test statistic
        
    precision: int, optional, default = 100
        The precision of the p-value calculation for the 'asymp' method.
        The p-value is calculated as 1 - kolmogorov_cdf(z, precision), where z is the test statistic.
        The default value is 100.

    verbose: bool, optional, default = False
        If True, print additional information.

    Returns: 
    --------
    result: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
        Tuple containing the following elements:

        - d: tf.Tensor, shape = (1,)
            KS test-statistic.

        - prob: tf.Tensor, shape = (1,)
            one- or two-sided p-value depending on the choice of 'alternative'.

        - d_location: tf.Tensor, shape = (1,)
            The x-location of the maximum difference of the cumulative distribution function.

        - d_sign: tf.Tensor, shape = (1,)
            The sign of the maximum difference of the cumulative distribution function.

    Note: 
    -----
    The 'exact' method is not (yet) implemented. 
    If selected, the function will automatically fall back to 'asymp'. 
    """
    alternative: int = alternative_int
    method: int = method_int
    mode: int = method

    def greatest_common_divisor_tf(x, y):
        while tf.not_equal(y, 0):
            x, y = y, tf.math.floormod(x, y)
        return x

    MAX_AUTO_N: tf.Tensor = tf.constant(10000, dtype=tf.float32) # type: ignore
    
    data1 = tf.sort(data1)
    data2 = tf.sort(data2)
    
    n1: tf.Tensor = tf.cast(tf.shape(data1)[0], tf.float32) # type: ignore
    n2: tf.Tensor = tf.cast(tf.shape(data2)[0], tf.float32) # type: ignore
    
    data_all: tf.Tensor = tf.concat([data1, data2], axis=0) # type: ignore

    # using searchsorted solves equal data problem
    cdf1: tf.Tensor = tf.cast(tf.searchsorted(data1, data_all, side = 'right'), tf.float32) / n1
    cdf2: tf.Tensor = tf.cast(tf.searchsorted(data2, data_all, side = 'right'), tf.float32) / n2
    cddiffs: tf.Tensor = cdf1 - cdf2
    
    # Identify the location of the statistic
    argminS: tf.Tensor = tf.argmin(cddiffs)
    argmaxS: tf.Tensor = tf.argmax(cddiffs)
    loc_minS: tf.Tensor = data_all[argminS]
    loc_maxS: tf.Tensor = data_all[argmaxS]
    
    # Ensure sign of minS is not negative.
    minS: tf.Tensor = tf.clip_by_value(-cddiffs[argminS], clip_value_min = 0, clip_value_max = 1) # type: ignore
    maxS: tf.Tensor = cddiffs[argmaxS]
    
    max_abs_diff: tf.Tensor = tf.maximum(minS, maxS) # type: ignore
    less_max: tf.Tensor = tf.greater_equal(minS, maxS) # type: ignore

    location: tf.Tensor = tf.where(less_max, loc_minS, loc_maxS)
    sign: tf.Tensor = tf.where(less_max, -1, 1)
    
    d: tf.Tensor = tf.where(tf.equal(alternative, 0), 
                            x = max_abs_diff, 
                            y = tf.where(tf.equal(alternative, 1), 
                                         x = minS, 
                                         y = maxS))
    d_location: tf.Tensor = tf.where(tf.equal(alternative, 0), 
                                     x = location, 
                                     y = tf.where(tf.equal(alternative, 1), 
                                                  x = loc_minS, 
                                                  y = loc_maxS))
    d_sign: tf.Tensor = tf.where(tf.equal(alternative, 0), 
                                 x = sign, 
                                 y = tf.where(tf.equal(alternative, 1), 
                                              x = -1, 
                                              y = 1))

    g: tf.Tensor = greatest_common_divisor_tf(n1, n2)
    n1g: tf.Tensor = tf.math.floordiv(n1, g) # type: ignore
    n2g: tf.Tensor = tf.math.floordiv(n2, g) # type: ignore
    prob: tf.Tensor = -tf.float32.max # type: ignore
        
    # If mode is 'auto' (0), decide between 'exact' (1) and 'asymp'  (2) based on n1, n2
    mode = tf.where(tf.equal(mode, 0),
                    x = tf.where(tf.less_equal(tf.reduce_max([n1, n2]), MAX_AUTO_N), 
                                 x = 1,
                                 y = 2),
                    y = mode)
    
    # If lcm(n1, n2) is too big, switch from 'exact' (1) to 'asymp' (2)
    mode = tf.where(tf.logical_and(tf.equal(mode, 1), tf.greater_equal(n1g, tf.int32.max / n2g)),
                    x = 2,
                    y = mode)

    # Exact calculation is not implemented, so switch from 'exact' (1) to 'asymp' (2)
    mode = tf.where(tf.equal(mode, 1), 
                    x = 2, 
                    y = mode)
    
    def asymp_ks_2samp(n1: tf.Tensor,
                       n2: tf.Tensor,
                       d: tf.Tensor,
                       alternative: int,
                       precision: int
                      ) -> Tuple[tf.Tensor, tf.Tensor]:
        sorted_values: tf.Tensor = tf.sort(tf.stack([tf.cast(n1, tf.float32), tf.cast(n2, tf.float32)]), direction='DESCENDING')
        m: tf.Tensor = sorted_values[0]
        n: tf.Tensor = sorted_values[1]
        en: tf.Tensor = m * n / (m + n)
        
        def kolmogorov_cdf(x: tf.Tensor, 
                           precision: int
                          ) -> tf.Tensor:
            k_values: tf.Tensor = tf.range(-precision, precision + 1, dtype=tf.float32)
            terms: tf.Tensor = (-1.)**k_values * tf.exp(-2. * k_values**2 * x**2)
            prob: tf.Tensor = tf.reduce_sum(terms)
            return prob
        
        def two_sided_p_value(d: tf.Tensor,
                              en: tf.Tensor,
                              precision: int
                             ) -> tf.Tensor:
            z: tf.Tensor = tf.sqrt(en) * d
            prob: tf.Tensor = 1 - kolmogorov_cdf(z, precision) # type: ignore
            return prob

        def one_sided_p_value() -> tf.Tensor:
            z = tf.sqrt(en) * d
            expt = -2 * z**2 - 2 * z * (m + 2*n)/tf.sqrt(m*n*(m+n))/3.0
            prob = tf.exp(expt)
            return prob
        
        prob = tf.where(tf.equal(alternative, 0), 
                        x = two_sided_p_value(d, en, precision), 
                        y = one_sided_p_value())

        return d, prob
    
    d, prob = asymp_ks_2samp(n1, n2, d, alternative, precision)

    prob = tf.clip_by_value(prob, 0, 1) # type: ignore
    
    return d, prob, d_location, d_sign


class KSTest(TwoSampleTestBase):
    """
    Class for computing the Kolmogorov-Smirnov test-statistic and p-value for two-sample tests.
    It inherits from the TwoSampleTestBase base class.
    The Kolmogorov-Smirnov test-statistic and p-value can be computed using either numpy or tensorflow.
    The scipy implementation is used for the numpy backend.
    A custom tensorflow implementation is used for the tensorflow backend.
    The tensorflow implementation is faster than the scipy implementation, especially for large dimensionality and 
    number of iterations.
    The tensorflow implementation only supports the 'asymp' method for computing the p-value.

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
    compute(max_vectorize: int = 100) -> None
        Compute the Kolmogorov-Smirnov test-statistic and p-value for two samples.
        If use_tf is True, the calculation is performed using tensorflow functions.
        Otherwise, the calculation is performed using numpy functions.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The calculation is parallelized over max_vectorize (out of ndims*niter).
        The results are stored in the Results attribute.

    Test_np() -> None
        Compute the Kolmogorov-Smirnov test-statistic and p-value for two samples using numpy functions.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.

    Test_tf(max_vectorize: int = 100) -> None
        Compute the Kolmogorov-Smirnov test-statistic and p-value for two samples using tensorflow functions.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The calculation is parallelized over max_vectorize (out of ndims*niter).
        The results are stored in the Results attribute.

    get_niter_batch_size_np() -> Tuple[int, int]
        Compute the number of iterations and the batch size for the numpy backend.
        The number of iterations is the smallest integer such that niter*batch_size >= nsamples.
        The batch size is the largest integer such that niter*batch_size <= max_vectorize.
        The number of iterations and the batch size are stored in the Inputs attribute.

    get_niter_batch_size_tf() -> Tuple[tf.Tensor, tf.Tensor]
        Compute the number of iterations and the batch size for the tensorflow backend.
        The number of iterations is the smallest integer such that niter*batch_size >= nsamples.
        The batch size is the largest integer such that niter*batch_size <= max_vectorize.
        The number of iterations and the batch size are stored in the Inputs attribute.

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

        # Compute KS test
        ks_test = GMetrics.KSTest(data_input = data_input, 
                                  progress_bar = True, 
                                  verbose = True)
        ks_test.compute(max_vectorize = 100)
        ks_test.Results[0].result_value
        
        >> {'statistic_lists': [[...]],
            'statistic_means': [...],
            'statistic_stds': [...],
            'pvalue_lists': [[...]],
            'pvalue_means': [...],
            'pvalue_stds': [...]}
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 progress_bar: bool = False,
                 verbose: bool = False
                ) -> None:
        """
        Class constructor.
        """
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
        Function that computes the Kolmogorov-Smirnov test-statistic and p-value for two samples
        selecting among the Test_np and Test_tf methods depending on the use_tf attribute.
        
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
        Function that computes the Kolmogorov-Smirnov test-statistic and p-value for two samples using numpy backend.
        The calculation is based in the scipy function scipy.stats.ks_2samp.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The results are stored in the Results attribute.
            
        Parameters:
        ----------
        None
            
        Returns:
        -------
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
            conditional_print(self.verbose, "Starting KS tests calculation...")
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
        
        statistic_lists: List[List] = []
        statistic_means: List[float] = []
        statistic_stds: List[float] = []
        pvalue_lists: List[List] = []
        pvalue_means: List[float] = []
        pvalue_stds: List[float] = []

        start_calculation()
        init_progress_bar()
            
        reset_random_seeds(seed = seed)
        
        conditional_print(self.verbose, "Running numpy KS tests...")
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
            list1: List[float] = []
            list2: List[float] = []
            for dim in range(ndims):
                statistic, pvalue = ks_2samp(dist_1_k[:,dim], dist_2_k[:,dim])
                list1.append(statistic)
                list2.append(pvalue)
            statistic_lists.append(list1)
            statistic_means.append(np.mean(list1)) # type: ignore
            statistic_stds.append(np.std(list1)) # type: ignore
            pvalue_lists.append(list2)
            pvalue_means.append(np.mean(list2)) # type: ignore
            pvalue_stds.append(np.std(list2)) # type: ignore
            update_progress_bar()
        
        close_progress_bar()
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "KS Test_np"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "numpy"}}
        result_value: Dict[str, DataTypeNP] = {"statistic_lists": np.array(statistic_lists),
                                               "statistic_means": np.array(statistic_means),
                                               "statistic_stds": np.array(statistic_stds),
                                               "pvalue_lists": np.array(pvalue_lists),
                                               "pvalue_means": np.array(pvalue_means),
                                               "pvalue_stds": np.array(pvalue_stds)}
        result = TwoSampleTestResult(timestamp, test_name, parameters, result_value) # type: ignore
        self.Results.append(result)
        
    def Test_tf(self, max_vectorize: int = 100) -> None:
        """
        Function that computes the Kolmogorov-Smirnov test-statistic and p-value for two samples using tensorflow functions.
        The calculation is based in the custom function ks_2samp_tf.
        The calculation is performed in batches of size batch_size.
        The number of batches is niter.
        The total number of samples is niter*batch_size.
        The calculation is parallelized over max_vectorize (out of ndims*niter).
        The results are stored in the Results attribute.

        Parameters:
        ----------
        max_vectorize: int, optional, default = 100
            A maximum number of batch_size*max_vectorize samples per time are processed by the tensorflow backend.
            Given a value of max_vectorize, the ndims*niter KS calculations are split in chunks of max_vectorize.
            Each chunk is processed by the tensorflow backend in parallel. If ndims is larger than max_vectorize,
            the calculation is vectorized niter times over ndims.

        Returns:
        -------
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
            conditional_tf_print(self.verbose, "Starting KS tests calculation...")
            conditional_tf_print(self.verbose, "Running TF KS tests...")
            conditional_tf_print(self.verbose, "niter =", niter)
            conditional_tf_print(self.verbose, "batch_size =", batch_size)
            self._start = timer()

        def end_calculation() -> None:
            self._end = timer()
            elapsed = self.end - self.start
            conditional_tf_print(self.verbose, "KS tests calculation completed in", str(elapsed), "seconds.")
            
        def set_dist_num_from_symb(dist: DistTypeTF,
                                   nsamples: int,
                                   seed: int = 0
                                  ) -> tf.Tensor:
            nonlocal dtype
            dist_num: tf.Tensor = tf.cast(dist.sample(nsamples, seed = int(seed)), dtype = dtype) # type: ignore
            return dist_num
        
        def return_dist_num(dist_num: tf.Tensor) -> tf.Tensor:
            return dist_num
            
        @tf.function(reduce_retracing=True)
        def batched_test(start, end):
            conditional_tf_print(tf.logical_and(tf.logical_or(tf.math.logical_not(tf.equal(start,0)),tf.math.logical_not(tf.equal(end,niter))), self.verbose), "Iterating from", start, "to", end, "out of", niter, ".")
            # Define unique constants for the two distributions. It is sufficient that these two are different to get different samples from the two distributions, if they are equal. 
            # There is not problem with subsequent calls to the batched_test function, since the random state is updated at each call.
            seed_dist_1  = int(1e6)  # Seed for distribution 1
            seed_dist_2  = int(1e12)  # Seed for distribution 2
            
            # Define batched distributions
            dist_1_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_1_symb, nsamples = batch_size*(end-start), seed = seed_dist_1),
                                               false_fn = lambda: return_dist_num(dist_1_num[start*batch_size:end*batch_size, :])) # type: ignore
            dist_2_k: tf.Tensor = tf.cond(tf.equal(tf.shape(dist_1_num[0])[0],0), # type: ignore
                                               true_fn = lambda: set_dist_num_from_symb(dist_2_symb, nsamples = batch_size*(end-start), seed = seed_dist_2),
                                               false_fn = lambda: return_dist_num(dist_2_num[start*batch_size:end*batch_size, :])) # type: ignore
            
            dist_1_k = tf.reshape(dist_1_k, (end-start, batch_size, ndims))
            dist_2_k = tf.reshape(dist_2_k, (end-start, batch_size, ndims))
                
            # Define the loop body function
            def loop_body(args):
                idx1 = args[0]
                idx2 = args[1]
                metric, pvalue, _, _ = ks_2samp_tf(dist_1_k[idx1, :, idx2], dist_2_k[idx1, :, idx2], verbose=False) # type: ignore
                metric = tf.cast(metric, dtype=dtype)
                pvalue = tf.cast(pvalue, dtype=dtype)
                return metric, pvalue
            
            # Create the range of indices for both loops
            indices = tf.stack(tf.meshgrid(tf.range(end-start), tf.range(ndims), indexing='ij'), axis=-1)
            indices = tf.reshape(indices, [-1, 2])
            
            # Use tf.vectorized_map to iterate over the indices
            statistic_lists, pvalue_lists = tf.vectorized_map(loop_body, indices) # type: ignore
            
            # Reshape the results back to (chunk_size, ndims)
            statistic_lists = tf.reshape(statistic_lists, (end-start, ndims))
            pvalue_lists = tf.reshape(pvalue_lists, (end-start, ndims))

            # Compute the mean values
            statistic_means = tf.cast(tf.reduce_mean(statistic_lists, axis=1), dtype=dtype)
            statistic_stds = tf.cast(tf.math.reduce_std(statistic_lists, axis=1), dtype=dtype)
            pvalue_means = tf.cast(tf.reduce_mean(pvalue_lists, axis=1), dtype=dtype)
            pvalue_stds = tf.cast(tf.math.reduce_std(pvalue_lists, axis=1), dtype=dtype)
            
            statistic_means = tf.expand_dims(statistic_means, axis=1)
            statistic_stds = tf.expand_dims(statistic_stds, axis=1)
            pvalue_means = tf.expand_dims(pvalue_means, axis=1)
            pvalue_stds = tf.expand_dims(pvalue_stds, axis=1)
            
            res = tf.concat([statistic_means, statistic_stds, statistic_lists, pvalue_means, pvalue_stds, pvalue_lists], axis=1)
        
            return res

        @tf.function(reduce_retracing=True)
        def compute_test(max_vectorize: int = 100) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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
            
            # Run the computation in chunks
            # Initialize the result TensorArray
            res = tf.TensorArray(dtype, size = nchunks)
            statistic_means = tf.TensorArray(dtype, size = nchunks)
            statistic_stds = tf.TensorArray(dtype, size = nchunks)
            statistic_lists = tf.TensorArray(dtype, size = nchunks)
            pvalue_means = tf.TensorArray(dtype, size = nchunks)
            pvalue_stds = tf.TensorArray(dtype, size = nchunks)
            pvalue_lists = tf.TensorArray(dtype, size = nchunks)

            def body(i, res):
                start = i * max_iter_per_chunk
                end = tf.minimum(start + max_iter_per_chunk, niter)
                chunk_result = batched_test(start, end) # type: ignore
                res = res.write(i, chunk_result)
                return i+1, res

            _, res = tf.while_loop(lambda i, res: i < nchunks, body, [0, res])
            
            for i in range(nchunks):
                res_i = res.read(i)
                statistic_means = statistic_means.write(i, res_i[:,0])
                statistic_stds = statistic_stds.write(i, res_i[:,1])
                statistic_lists = statistic_lists.write(i, res_i[:,2:2+ndims])
                pvalue_means = pvalue_means.write(i, res_i[:,2+ndims])
                pvalue_stds = pvalue_stds.write(i, res_i[:,3+ndims])
                pvalue_lists = pvalue_lists.write(i, res_i[:,4+ndims:])
                
            statistic_means_stacked = tf.reshape(statistic_means.stack(), (niter,))
            statistic_stds_stacked = tf.reshape(statistic_stds.stack(), (niter,))
            statistic_lists_stacked = tf.reshape(statistic_lists.stack(), (niter, ndims))
            pvalue_means_stacked = tf.reshape(pvalue_means.stack(), (niter,))
            pvalue_stds_stacked = tf.reshape(pvalue_stds.stack(), (niter,))
            pvalue_lists_stacked = tf.reshape(pvalue_lists.stack(), (niter, ndims))
            
            return statistic_means_stacked, statistic_stds_stacked, statistic_lists_stacked, pvalue_means_stacked, pvalue_stds_stacked, pvalue_lists_stacked
                
        start_calculation()
        
        reset_random_seeds(seed = seed)
        
        statistic_means, statistic_stds, statistic_lists, pvalue_means, pvalue_stds, pvalue_lists = compute_test(max_vectorize = max_vectorize) # type: ignore
                             
        end_calculation()
        
        timestamp: str = datetime.now().isoformat()
        test_name: str = "KS Test_np"
        parameters: Dict[str, Any] = {**self.param_dict, **{"backend": "tensorflow"}}
        result_value: Dict[str, DataTypeNP] = {"statistic_lists": statistic_lists.numpy(),
                                               "statistic_means": statistic_means.numpy(),
                                               "statistic_stds": statistic_stds.numpy(),
                                               "pvalue_lists": pvalue_lists.numpy(),
                                               "pvalue_means": pvalue_means.numpy(),
                                               "pvalue_stds": pvalue_stds.numpy()}
        result = TwoSampleTestResult(timestamp, test_name, parameters, result_value) # type: ignore
        self.Results.append(result)