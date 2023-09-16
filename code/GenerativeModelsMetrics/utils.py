__all__ = [
    'reset_random_seeds',
    'NumpyDistribution',
    'get_best_dtype_np',
    'get_best_dtype_tf',
    'conditional_print',
    'conditional_tf_print',
    'parse_input_dist_np',
    'parse_input_dist_tf'
]
import os
import inspect
import numpy as np
import random
from scipy.stats import moment # type: ignore
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from numpy import random as npr
from timeit import default_timer as timer

from typing import Tuple, Union, Optional, Type, Callable, Dict, List
from numpy import typing as npt
# For future tensorflow typing support
#import tensorflow.python.types.core as tft
#IntTensor = tft.Tensor[tf.int32]
#FloatTensor = Union[tft.Tensor[tf.float32], tft.Tensor[tf.float64]]
#BoolTensor = Type[tft.Tensor[tf.bool]]
DTypeType = Union[tf.DType, np.dtype, type]
IntTensor = Type[tf.Tensor]
FloatTensor = Type[tf.Tensor]
BoolTypeTF = Type[tf.Tensor]
BoolTypeNP = np.bool_
IntType = Union[int, IntTensor]
DataTypeTF = FloatTensor
DataTypeNP = npt.NDArray[np.float_]
DataType = Union[DataTypeNP, DataTypeTF]
DistTypeTF = tfp.distributions.Distribution

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    npr.seed(seed)
    random.seed(seed)

class NumpyDistribution:
    """
    Wrapper class for numpy.random.Generator distributions.
    Example:

    .. code-block:: python
    
        dist = NumpyDistribution('normal', loc=0, scale=1)
        dist.sample(100)
        
    """
    def __init__(self, 
                 distribution: str = "standard_normal",
                 generator_input: np.random.Generator = np.random.default_rng(),
                 dtype: type = np.float32,
                 **kwargs):
        self.generator: np.random.Generator = generator_input
        self.distribution: str = distribution
        self.dtype: type = dtype
        self.params: dict = kwargs

        # Check if the distribution is a valid method of the generator
        if not self.is_valid_distribution():
            raise ValueError(f"{distribution} is not a valid distribution of numpy.random.Generator.")

    def is_valid_distribution(self) -> bool:
        """Check if the given distribution is a valid method of the generator."""
        return self.distribution in dir(self.generator)

    def sample(self, 
               n: int,
               seed: Optional[int] = None
               ) -> DataTypeNP:
        """Generate a sample from the distribution."""
        if seed:
            reset_random_seeds(seed = seed)
        method: Callable = getattr(self.generator, self.distribution)
        if inspect.isroutine(method):
            return method(size=n, **self.params).astype(self.dtype)
        else:
            raise ValueError(f"{self.distribution} is not a callable method of numpy.random.Generator.")

DistTypeNP = NumpyDistribution
DistType = Union[tfp.distributions.Distribution, DistTypeNP]
DataDistTypeNP = Union[DataTypeNP, DistTypeNP]
DataDistTypeTF = Union[DataTypeTF, tfp.distributions.Distribution]
DataDistType = Union[DataDistTypeNP, DataDistTypeTF]
BoolType = Union[bool, BoolTypeTF, BoolTypeNP]
  
def compute_lik_ratio_statistic(dist_ref: tfp.distributions.Distribution,
                                dist_alt: tfp.distributions.Distribution,
                                sample_ref: tf.Tensor,
                                sample_alt: tf.Tensor,
                                batch_size: int = 10000
                               ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    start_global = timer()
    
    # Compute number of samples
    n_ref = len(sample_ref)
    n_alt = len(sample_alt)
    
    print(f"Computing likelihood ratio statistic with {n_ref} reference samples and {n_alt} alternative samples.")
    
    # Compute log probabilities without reshaping
    start = timer()
    logprob_ref_ref = tf.reshape(dist_ref.log_prob(sample_ref), [-1, batch_size])
    end = timer()
    print(f"Computed logprob of ref dist on ref samples in {end - start:.2f} s.")
    start = timer()
    logprob_ref_alt = tf.reshape(dist_ref.log_prob(sample_alt), [-1, batch_size])
    end = timer()
    print(f"Computed logprob of ref dist on alt samples in {end - start:.2f} s.")
    start = timer()
    logprob_alt_alt = tf.reshape(dist_alt.log_prob(sample_alt), [-1, batch_size])
    end = timer()
    print(f"Computed logprob of alt dist on alt samples in {end - start:.2f} s.")
    
    # Reshape the log probabilities
    logprob_ref_ref_reshaped = tf.reshape(logprob_ref_ref, [-1, batch_size])
    logprob_ref_alt_reshaped = tf.reshape(logprob_ref_alt, [-1, batch_size])
    logprob_alt_alt_reshaped = tf.reshape(logprob_alt_alt, [-1, batch_size])
    
    # Create masks for finite log probabilities
    finite_indices_ref_ref = tf.math.is_finite(logprob_ref_ref_reshaped)
    finite_indices_ref_alt = tf.math.is_finite(logprob_ref_alt_reshaped)
    finite_indices_alt_alt = tf.math.is_finite(logprob_alt_alt_reshaped)
    
    # Count the number of finite samples
    n_ref_finite = tf.reduce_sum(tf.cast(finite_indices_ref_ref, tf.int32))
    n_alt_finite = tf.reduce_sum(tf.cast(tf.math.logical_and(finite_indices_alt_alt, finite_indices_ref_alt), tf.int32))

    if n_ref_finite < n_ref:
        fraction = tf.cast(n_ref - n_ref_finite, tf.float32) / tf.cast(n_ref, tf.float32) # type: ignore
        print(f"Warning: Removed a fraction {fraction} of reference samples due to non-finite log probabilities.")
        
    if n_alt_finite < n_alt:
        fraction = tf.cast(n_alt - n_alt_finite, tf.float32) / tf.cast(n_alt, tf.float32) # type: ignore
        print(f"Warning: Removed a fraction {fraction} of alternative samples due to non-finite log probabilities.")
    
    # Combined finite indices
    combined_finite_indices = tf.math.logical_and(tf.math.logical_and(finite_indices_ref_ref, finite_indices_ref_alt), finite_indices_alt_alt)
    
    # Use masks to filter the reshaped log probabilities
    logprob_ref_ref_filtered = tf.where(combined_finite_indices, logprob_ref_ref_reshaped, 0.)
    logprob_ref_alt_filtered = tf.where(combined_finite_indices, logprob_ref_alt_reshaped, 0.)
    logprob_alt_alt_filtered = tf.where(combined_finite_indices, logprob_alt_alt_reshaped, 0.)
    
    ## Filter the log probabilities using the mask
    #logprob_ref_ref_filtered = tf.boolean_mask(logprob_ref_ref_reshaped, combined_finite_indices)
    #logprob_ref_alt_filtered = tf.boolean_mask(logprob_ref_alt_reshaped, combined_finite_indices)
    #logprob_alt_alt_filtered = tf.boolean_mask(logprob_alt_alt_reshaped, combined_finite_indices)
    
    # Compute log likelihoods
    logprob_ref_ref_sum = tf.reduce_sum(logprob_ref_ref_filtered, axis=1)
    logprob_ref_alt_sum = tf.reduce_sum(logprob_ref_alt_filtered, axis=1)
    logprob_alt_alt_sum = tf.reduce_sum(logprob_alt_alt_filtered, axis=1)
    lik_ref_dist = logprob_ref_ref_sum + logprob_ref_alt_sum
    lik_alt_dist = logprob_ref_ref_sum + logprob_alt_alt_sum
    
    # Compute likelihood ratio statistic
    lik_ratio = 2 * (lik_alt_dist - lik_ref_dist)
    print(f'lik_ratio = {lik_ratio}')
    
    # Casting to float32 before performing division
    n_ref_finite_float = tf.cast(n_ref_finite, tf.float32)
    n_alt_finite_float = tf.cast(n_alt_finite, tf.float32)  

    # Compute normalized likelihood ratio statistic
    n = 2 * n_ref_finite_float * n_alt_finite_float / (n_ref_finite_float + n_alt_finite_float)
    
    # Compute normalized likelihood ratio statistic
    lik_ratio_norm = lik_ratio / tf.sqrt(tf.cast(n, tf.float32))

    end_global = timer()
    
    print(f"Computed likelihood ratio statistic in {end_global - start_global:.2f} s.")
    
    return logprob_ref_ref_sum, logprob_ref_alt_sum, logprob_alt_alt_sum, lik_ratio, lik_ratio_norm

    
def get_best_dtype_np(dtype_1: Union[type, np.dtype],
                      dtype_2: Union[type, np.dtype]) -> Union[type, np.dtype]:
    dtype_1_precision = np.finfo(dtype_1).eps
    dtype_2_precision = np.finfo(dtype_2).eps
    if dtype_1_precision > dtype_2_precision:
        return dtype_1
    else:
        return dtype_2
    
def get_best_dtype_tf(dtype_1: tf.DType, 
                      dtype_2: tf.DType) -> tf.DType:
    dtype_1_precision: tf.Tensor = tf.abs(tf.as_dtype(dtype_1).min)
    dtype_2_precision: tf.Tensor = tf.abs(tf.as_dtype(dtype_2).min)

    dtype_out = tf.cond(tf.greater(dtype_1_precision, dtype_2_precision), 
                        lambda: dtype_1,
                        lambda: dtype_2)
    return tf.as_dtype(dtype_out)

def conditional_print(verbose: bool = False,
                      *args) -> None:
    if verbose:
        print(*args)

#@tf.function(reduce_retracing = True)
def conditional_tf_print(verbose: bool = False,
                         *args) -> None:
    tf.cond(tf.equal(verbose, True), lambda: tf.print(*args), lambda: verbose)

def parse_input_dist_np(dist_input: DataDistTypeNP,
                        verbose: bool = False
                       ) -> Tuple[bool, DistTypeNP, DataTypeNP, int, int]:
    dist_symb: DistTypeNP
    dist_num: DataTypeNP
    nsamples: int
    ndims: int
    is_symb: bool
    if verbose:
        print("Parsing input distribution...")
    if isinstance(dist_input, np.ndarray):
        if verbose:
            print("Input distribution is a numberic numpy array or tf.Tensor")
        if len(dist_input.shape) != 2:
            raise ValueError("Input must be a 2-dimensional numpy array or a tfp.distributions.Distribution object")
        else:
            dist_symb = NumpyDistribution()
            dist_num = dist_input
            nsamples, ndims = dist_num.shape
            is_symb = False
    elif isinstance(dist_input, NumpyDistribution):
        if verbose:
            print("Input distribution is a NumpyDistribution object.")
        dist_symb = dist_input
        dist_num = np.array([[]],dtype=dist_symb.dtype)
        nsamples, ndims = 0, dist_symb.sample(2).shape[1]
        is_symb = True
    else:
        raise ValueError("Input must be either a numpy array or a NumpyDistribution object.")
    return is_symb, dist_symb, dist_num, ndims, nsamples


def parse_input_dist_tf(dist_input: DataDistType,
                        verbose: bool = False
                       ) -> Tuple[BoolType, DistTypeTF, DataTypeTF, IntType, IntType]:
    
    def is_ndarray_or_tensor():
        return tf.reduce_any([isinstance(dist_input, np.ndarray), tf.is_tensor(dist_input)])
    
    def is_distribution():
        return tf.reduce_all([
            tf.logical_not(is_ndarray_or_tensor()),
            tf.reduce_any([isinstance(dist_input, tfp.distributions.Distribution)])
        ])

    def handle_distribution():
        conditional_tf_print(verbose, "Input distribution is a tfp.distributions.Distribution object.")
        dist_symb: tfp.distributions.Distribution = dist_input
        nsamples, ndims = tf.constant(0), tf.shape(dist_symb.sample(2))[1]
        dist_num = tf.convert_to_tensor([[]],dtype=dist_symb.dtype)
        return tf.constant(True), dist_symb, dist_num, ndims, nsamples

    def handle_ndarray_or_tensor():
        conditional_tf_print(verbose, "Input distribution is a numeric numpy array or tf.Tensor.")
        if tf.rank(dist_input) != 2:
            tf.debugging.assert_equal(tf.rank(dist_input), 2, "Input must be a 2-dimensional numpy array or a tfp.distributions.Distribution object.")
        dist_symb = tfp.distributions.Normal(loc=tf.zeros(dist_input.shape[1]), scale=tf.ones(dist_input.shape[1])) # type: ignore
        dist_num = tf.convert_to_tensor(dist_input)
        nsamples, ndims = tf.unstack(tf.shape(dist_num))
        return tf.constant(False), dist_symb, dist_num, ndims, nsamples

    def handle_else():
        tf.debugging.assert_equal(
            tf.reduce_any([is_distribution(), is_ndarray_or_tensor()]),
            True,
            "Input must be either a numpy array or a tfp.distributions.Distribution object."
        )

    conditional_tf_print(verbose, "Parsing input distribution...")

    return tf.case([
        (is_distribution(), handle_distribution),
        (is_ndarray_or_tensor(), handle_ndarray_or_tensor)
    ], default=handle_else, exclusive=True)


def se_mean(data):
    n = len(data)
    mu_2 = moment(data, moment=2)  # second central moment (variance)
    se_mean = mu_2 / np.sqrt(n)
    return se_mean

def se_std(data):
    n = len(data)
    mu_2 = moment(data, moment=2)  # second central moment (variance)
    mu_4 = moment(data, moment=4)  # fourth central moment
    se_std = np.sqrt((mu_4 - mu_2**2) / (4 * mu_2 * n))
    return se_std

def generate_and_clean_data_simple(dist, n_samples, batch_size, dtype, seed):
    if dtype is None:
        dtype = tf.float32
    X_data = []
    total_samples = 0

    while total_samples < n_samples:
        try:
            batch = dist.sample(batch_size, seed=seed)

            # Find finite values
            finite_indices = tf.reduce_all(tf.math.is_finite(batch), axis=1)

            # Warn the user if there are any non-finite values
            n_nonfinite = tf.reduce_sum(tf.cast(~finite_indices, tf.int32))
            if n_nonfinite > 0:
                print(f"Warning: Removed {n_nonfinite} non-finite values from the batch")

            # Select only the finite values
            finite_batch = batch.numpy()[finite_indices.numpy()].astype(dtype.as_numpy_dtype)

            X_data.append(finite_batch)
            total_samples += len(finite_batch)
        except (RuntimeError, tf.errors.ResourceExhaustedError):
            # If a RuntimeError or a ResourceExhaustedError occurs (possibly due to OOM), halve the batch size
            batch_size = batch_size // 2
            print("Warning: Batch size too large. Halving batch size to {}".format(batch_size),"and retrying.")
            if batch_size == 0:
                raise RuntimeError("Batch size is zero. Unable to generate samples.")

    return np.concatenate(X_data, axis=0)[:n_samples]

def generate_and_clean_data_mirror(dist, n_samples, batch_size, dtype, seed):
    if dtype is None:
        dtype = tf.float32
        
    strategy = tf.distribute.MirroredStrategy()  # setup the strategy

    # Create a new random generator with a seed
    rng = tf.random.Generator.from_seed(seed)

    # Compute the global batch size using the number of replicas.
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    @tf.function
    def sample_and_clean(dist, batch_size, rng):
        new_seed = rng.make_seeds(2)[0]
        new_seed = tf.cast(new_seed, tf.int32)  # Convert the seed to int32
        batch = dist.sample(batch_size, seed=new_seed)
        finite_indices = tf.reduce_all(tf.math.is_finite(batch), axis=1)
        finite_batch = tf.boolean_mask(batch, finite_indices)
        return finite_batch, tf.shape(finite_batch)[0]

    total_samples = 0
    samples = []

    with strategy.scope():
        while total_samples < n_samples:
            try:
                per_replica_samples, per_replica_sample_count = strategy.run(sample_and_clean, args=(dist, global_batch_size, rng))
                # concatenate samples on each replica and append to list
                per_replica_samples_concat = tf.concat(strategy.experimental_local_results(per_replica_samples), axis=0)
                samples.append(per_replica_samples_concat)
                total_samples += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_sample_count, axis=None)
                #print("Generated {} samples".format(total_samples))
            except (RuntimeError, tf.errors.ResourceExhaustedError):
                # If a RuntimeError or a ResourceExhaustedError occurs (possibly due to OOM), halve the batch size
                global_batch_size = global_batch_size // 2
                print("Warning: Batch size too large. Halving batch size to {}".format(global_batch_size),"and retrying.")
                if global_batch_size == 0:
                    raise RuntimeError("Batch size is zero. Unable to generate samples.")

    # concatenate all samples
    samples = tf.concat(samples, axis=0)

    # return the first `n_samples` samples
    return samples[:n_samples]

def generate_and_clean_data(dist, n_samples, batch_size, dtype, seed, mirror_strategy = False):
    if batch_size > n_samples:
        batch_size = n_samples
        print("Warning: Batch size larger than number of samples. Setting batch size to {}".format(batch_size)) 
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if mirror_strategy:
        if len(gpu_devices) > 1:
            return generate_and_clean_data_mirror(dist, n_samples, batch_size, dtype, seed)
        else:
            return generate_and_clean_data_simple(dist, n_samples, batch_size, dtype, seed)
    else:
        return generate_and_clean_data_simple(dist, n_samples, batch_size, dtype, seed)