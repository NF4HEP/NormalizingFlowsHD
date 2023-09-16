__all__ = [
    'TwoSampleTestInputs',
    'TwoSampleTestResult',
    'TwoSampleTestBase'
]

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd # type: ignore
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import traceback
from timeit import default_timer as timer
from tqdm import tqdm # type: ignore
from . import utils
from .utils import reset_random_seeds
from .utils import parse_input_dist_np
from .utils import parse_input_dist_tf
from .utils import get_best_dtype_np
from .utils import get_best_dtype_tf
from .utils import generate_and_clean_data
from .utils import NumpyDistribution
from dataclasses import dataclass

from typing import Tuple, Union, Optional, Type, Dict, Any, List
from .utils import DTypeType, IntTensor, FloatTensor, BoolTypeTF, BoolTypeNP, IntType, DataTypeTF, DataTypeNP, DataType, DistTypeTF, DistTypeNP, DistType, DataDistTypeNP, DataDistTypeTF, DataDistType, BoolType

class TwoSampleTestInputs(object):
    """
    Class for validating data.
    """
    def __init__(self, 
                 dist_1_input: DataDistType,
                 dist_2_input: DataDistType,
                 niter: int = 10,
                 batch_size: int = 100000,
                 dtype_input: DTypeType = np.float32,
                 seed_input: Optional[int] = None,
                 use_tf: bool = False,
                 mirror_strategy: bool = False,
                 verbose: bool = False,
                ) -> None:
        # Attributes from arguments
        self._dist_1_input: DataDistType = dist_1_input
        self._dist_2_input: DataDistType = dist_2_input
        self._niter: int = niter
        self._batch_size: int = batch_size
        self._dtype_input: tf.DType = dtype_input
        self._seed: int = seed_input or int(np.random.randint(0, 2**32 - 1))
        self._use_tf: bool = use_tf
        self.verbose: bool = verbose
        
        # Attributes from preprocessing
        self._is_symb_1: BoolType
        self._is_symb_2: BoolType
        self._dist_1_symb: DistType
        self._dist_2_symb: DistType
        self._dist_1_num: DataType
        self._dist_2_num: DataType
        self._ndims_1: IntType
        self._ndims_2: IntType
        self._ndims: int
        self._nsamples_1: IntType
        self._nsamples_2: IntType
        self._nsamples: int
        self._dtype: DTypeType
        self._small_sample: bool
        
        # Preprocessing
        self.__preprocess(mirror_strategy = mirror_strategy, verbose = verbose)
            
    @property
    def dist_1_input(self) -> DataDistType:
        return self._dist_1_input
    
    @dist_1_input.setter
    def dist_1_input(self, 
                     dist_1_input: DataDistType
                    ) -> None:
        if isinstance(dist_1_input, (np.ndarray, NumpyDistribution, tf.Tensor, tfd.Distribution)):
            self._dist_1_input = dist_1_input
        else:
            raise TypeError("dist_1_input must be a np.ndarray, NumpyDistribution, tf.Tensor, or tfd.Distribution")
        self.__preprocess(verbose = False)
        
    @property
    def dist_2_input(self) -> DataDistType:
        return self._dist_2_input

    @dist_2_input.setter
    def dist_2_input(self,
                     dist_2_input: DataDistType
                    ) -> None:
        if isinstance(dist_2_input, (np.ndarray, NumpyDistribution, tf.Tensor, tfd.Distribution)):
            self._dist_2_input = dist_2_input
        else:   
            raise TypeError("dist_2_input must be a np.ndarray, NumpyDistribution, tf.Tensor, or tfd.Distribution")
        self.__preprocess(verbose = False)
            
    @property
    def niter(self) -> int:
        return self._niter
            
    @niter.setter
    def niter(self,
              niter: int
              ) -> None:
        if isinstance(niter, int):
            if niter > 0:
                self._niter = niter
            else:
                raise ValueError("niter must be positive")
        else:
            raise TypeError("niter must be an int")
        self.__preprocess(verbose = False)

    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self,
                   batch_size: int
                  ) -> None:
        if isinstance(batch_size, int):
            if batch_size > 0:
                self._batch_size = batch_size
            else:
                raise ValueError("batch_size must be positive")
        self.__preprocess(verbose = False)
        
    @property
    def dtype_input(self) -> Union[tf.DType, np.dtype, type]:
        return self._dtype_input

    @dtype_input.setter
    def dtype_input(self,
                    dtype_input: Union[tf.DType, np.dtype, type]
                   ) -> None:
        if isinstance(dtype_input, (tf.DType, np.dtype, type)):
            self._dtype_input = dtype_input
        else:
            raise TypeError("dtype_input must be a tf.DType, np.dtype, or type")
        self.__preprocess(verbose = False)
        
    @property
    def seed(self) -> int:
        return self._seed
    
    @seed.setter
    def seed(self,
             seed_input: int
            ) -> None:
        if isinstance(seed_input, int):
            self._seed = seed_input
        else:
            raise TypeError("seed_input must be an int")
        self.__preprocess(verbose = False)
        
    @property
    def use_tf(self) -> bool:
        return self._use_tf

    @use_tf.setter
    def use_tf(self,
                use_tf: BoolType
                ) -> None:
        if isinstance(use_tf, (bool,np.bool_)):
            self._use_tf = bool(use_tf)
        elif isinstance(use_tf, tf.Tensor):
            if isinstance(use_tf.numpy(), np.bool_):
                self._use_tf = bool(use_tf)
            else:
                raise TypeError("use_tf must be a bool or tf.Tensor with numpy() method returning a np.bool_")
        else:
            raise TypeError("use_tf must be a bool or tf.Tensor with numpy() method returning a np.bool_")
        self.__preprocess(verbose = False)
        
    @property
    def verbose(self) -> bool: # type: ignore
        return self._verbose

    @verbose.setter
    def verbose(self, # type: ignore
                verbose: Union[int,bool]
                ) -> None:
        if isinstance(verbose, bool):
            self._verbose = verbose
        elif isinstance(verbose, int):
            self._verbose = bool(verbose)
        else:
            raise TypeError("verbose must be a bool or an int (which is automatically converted to a bool)")
            
    @property
    def is_symb_1(self) -> BoolType:
        return self._is_symb_1
    
    @property
    def is_symb_2(self) -> BoolType:
        return self._is_symb_2
            
    @property
    def dist_1_symb(self) -> DistType:
        return self._dist_1_symb
    
    @property
    def dist_2_symb(self) -> DistType:
        return self._dist_2_symb
    
    @property
    def dist_1_num(self) -> DataType:
        return self._dist_1_num

    @property
    def dist_2_num(self) -> DataType:
        return self._dist_2_num
    
    @property
    def ndims_1(self) -> IntType:
        return self._ndims_1
    
    @property
    def ndims_2(self) -> IntType:
        return self._ndims_2
    
    @property
    def ndims(self) -> int:
        return self._ndims

    @property
    def nsamples_1(self) -> IntType:
        return self._nsamples_1

    @property
    def nsamples_2(self) -> IntType:
        return self._nsamples_2
    
    @property
    def nsamples(self) -> int:
        return self._nsamples
        
    @property
    def dtype_1(self) -> DTypeType:
        try:
            if isinstance(self.dist_1_num, (np.ndarray,tf.Tensor)):
                return self.dist_1_num.dtype
            else:
                raise AttributeError("dist_1_num should be a np.ndarray or tf.Tensor")
        except AttributeError:
            raise AttributeError("dist_1_num should be a np.ndarray or tf.Tensor")
            
    @property
    def dtype_2(self) -> DTypeType:
        try:
            if isinstance(self.dist_2_num, (np.ndarray,tf.Tensor)):
                return self.dist_2_num.dtype
            else:
                raise AttributeError("dist_2_num should be a np.ndarray or tf.Tensor")
        except AttributeError:
            raise AttributeError("dist_2_num should be a np.ndarray or tf.Tensor")

    @property
    def dtype(self) -> DTypeType:
        return self._dtype
    
    @property
    def small_sample(self) -> bool:
        return self._small_sample
    
    @small_sample.setter
    def small_sample(self,
                     small_sample: BoolType
                    ) -> None:
        if isinstance(small_sample, (bool,np.bool_)):
            self._small_sample = bool(small_sample)
        elif isinstance(small_sample, tf.Tensor):
            if isinstance(small_sample.numpy(), np.bool_):
                self._small_sample = bool(small_sample)
            else:
                raise TypeError("small_sample must be a bool or tf.Tensor with numpy() method returning a np.bool_")
        else:
            raise TypeError("small_sample must be a bool or tf.Tensor with numpy() method returning a np.bool_")
        self.__check_set_distributions()
        if self.small_sample:
            if not (self.batch_size*self.niter*self.ndims <= 1e7 or self.nsamples*self.ndims <= 1e7):
                print("Warning: small_sample is set to True, but the number of samples is large. This may cause memory issues.")
        if not self.small_sample:
            if self.batch_size*self.niter*self.ndims <= 1e7 or self.nsamples*self.ndims <= 1e7:
                print("Warning: small_sample is set to False, but the number of samples is small. Setting small_sample to True may speed up calculations.")
                        
    def __parse_input_dist(self,
                           dist_input: DataDistType,
                           verbose: bool = False
                          ) -> Tuple[BoolType, DistType, DataType, IntType, IntType]:
        if isinstance(dist_input, NumpyDistribution):
            if self.use_tf:
                if self.verbose:
                    print("To use tf mode, please use tf distributions or numerical tensors/arrays.")
                self.use_tf = False
        if self.use_tf:
            if isinstance(dist_input, (tf.Tensor, tfp.distributions.Distribution)):
                return parse_input_dist_tf(dist_input = dist_input, verbose = verbose)
            elif isinstance(dist_input, (np.ndarray, NumpyDistribution)):
                if self.verbose:
                    print("To use tf mode, please use tf distributions or numerical tensors/arrays.")
                self.use_tf = False
                return parse_input_dist_np(dist_input = dist_input, verbose = verbose)
            else:
                raise TypeError("dist_input must be a tf.Tensor, tfp.distributions.Distribution, np.ndarray, or NumpyDistribution")
        else:
            if isinstance(dist_input, (tf.Tensor, tfp.distributions.Distribution)):
                if self.verbose:
                    print("Using numpy mode with TensorFlow inputs.")
                return parse_input_dist_tf(dist_input = dist_input, verbose = verbose)
            elif isinstance(dist_input, (np.ndarray, NumpyDistribution)):
                return parse_input_dist_np(dist_input = dist_input, verbose = verbose)
            else:
                raise TypeError("dist_input must be a tf.Tensor, tfp.distributions.Distribution, np.ndarray, or NumpyDistribution")
        
    def __get_best_dtype(self,
                         dtype_1: DTypeType,
                         dtype_2: DTypeType,
                        ) -> DTypeType:   
        if self.use_tf:
            dtype_1 = tf.as_dtype(dtype_1)
            dtype_2 = tf.as_dtype(dtype_2)
            return get_best_dtype_tf(dtype_1 = dtype_1, dtype_2 = dtype_2)
        else:
            if isinstance(dtype_1, tf.DType):
                dtype_1 = dtype_1.as_numpy_dtype
            if isinstance(dtype_2, tf.DType):
                dtype_2 = dtype_2.as_numpy_dtype
            return get_best_dtype_np(dtype_1 = dtype_1, dtype_2 = dtype_2)
        
    def __parse_input_distributions(self,
                                    verbose: bool = False
                                    ) -> None:
        self._is_symb_1, self._dist_1_symb, self._dist_1_num, self._ndims_1, self._nsamples_1 = self.__parse_input_dist(dist_input = self._dist_1_input, verbose = verbose)
        self._is_symb_2, self._dist_2_symb, self._dist_2_num, self._ndims_2, self._nsamples_2 = self.__parse_input_dist(dist_input = self._dist_2_input, verbose = verbose)
        
    def __check_set_dtype(self) -> None:
        self._dtype = self.__get_best_dtype(self.dtype_input, self.__get_best_dtype(self.dtype_1, self.dtype_2))
    
    def __check_set_ndims_np(self) -> None:
        self._ndims_1 = int(self.ndims_1) # type: ignore
        self._ndims_2 = int(self.ndims_2) # type: ignore 
        if not (isinstance(self.ndims_1, int) and isinstance(self.ndims_2, int)):
            raise ValueError("ndims_1 and ndims_2 should be integers when in 'numpy' mode.")
        if self.ndims_1 != self.ndims_2:
            raise ValueError("dist_1 and dist_2 must have the same number of dimensions")
        else:
            self._ndims = self.ndims_1
            
    def __check_set_ndims_tf(self) -> None:
        # Utility functions
        def set_ndims(value: IntType) -> None:
            self._ndims = int(tf.constant(value))

        def raise_non_integer(value: IntType) -> None:
            value_tf = tf.constant(value)
            tf.debugging.assert_equal(value_tf.dtype.is_integer, tf.constant(True), message="non integer dimensions")
            
        def raise_none_equal_dims_error(value1: Any, value2: Any) -> None:
            tf.debugging.assert_equal(value1, value2, message="dist_1 and dist_2 must have the same number of dimensions")
        
        raise_non_integer(self.ndims_1)
        raise_non_integer(self.ndims_2)
        tf.cond(tf.equal(self.ndims_1, self.ndims_2), 
                true_fn = lambda: set_ndims(self.ndims_1), 
                false_fn = lambda: raise_none_equal_dims_error(self.ndims_1, self.ndims_2))
            
    def __check_set_ndims(self) -> None:
        if self.use_tf:
            self.__check_set_ndims_tf()
        else:
            self.__check_set_ndims_np()
            
    def __check_set_nsamples_np(self) -> None:
        if not (isinstance(self.nsamples_1, int) and isinstance(self.nsamples_2, int)):
            raise ValueError("nsamples_1 and nsamples_2 should be integers when in 'numpy' mode.")
        if self.nsamples_1 != 0 and self.nsamples_2 != 0:
            self._nsamples = np.minimum(int(self.nsamples_1), int(self.nsamples_2))
        elif self.nsamples_1 != 0 and self.nsamples_2 == 0:
            self._nsamples = int(self.nsamples_1)
        elif self.nsamples_1 == 0 and self.nsamples_2 != 0:
            self._nsamples = int(self.nsamples_2)
        elif self.nsamples_1 == 0 and self.nsamples_2 == 0:
            self._nsamples = int(self.batch_size * self.niter)
        else:
            raise ValueError("nsamples_1 and nsamples_2 should be positive integers or zero.")
        
    def __check_set_nsamples_tf(self) -> None:
        # Utility functions
        def set_nsamples(value: IntType) -> None:
            self._nsamples = int(tf.constant(value))
        nsamples_1 = int(tf.constant(self.nsamples_1))
        nsamples_2 = int(tf.constant(self.nsamples_2))
        nsamples_min = int(tf.constant(tf.minimum(nsamples_1, nsamples_2)))
        tf.cond(tf.not_equal(self.nsamples_1, tf.constant(0)),
                true_fn = lambda: tf.cond(tf.not_equal(self.nsamples_2, tf.constant(0)),
                                          true_fn = lambda: set_nsamples(nsamples_min),
                                          false_fn = lambda: set_nsamples(self.nsamples_1)),
                false_fn = lambda: tf.cond(tf.not_equal(self.nsamples_2, tf.constant(0)),
                                           true_fn = lambda: set_nsamples(self.nsamples_2),
                                           false_fn = lambda: set_nsamples(self.batch_size * self.niter)))
        
    def __check_set_nsamples(self) -> None:
        if self.use_tf:
            self.__check_set_nsamples_tf()
        else:
            self.__check_set_nsamples_np()
            
    def __check_set_small_sample_np(self) -> None:
        if self.batch_size*self.niter*self.ndims <= 1e7 or self.nsamples*self.ndims <= 1e7:
            self.small_sample = True
        else:
            self.small_sample = False
    
    def __check_set_small_sample_tf(self) -> None:
        # Utility functions
        def set_small_sample(value: bool) -> None:
            self.small_sample = value
        tf.cond(tf.logical_or(tf.less_equal(self.batch_size*self.niter*self.ndims, tf.constant(1e7)), 
                              tf.less_equal(self.nsamples*self.ndims, tf.constant(1e7))),
                true_fn = lambda: set_small_sample(True),
                false_fn = lambda: set_small_sample(False))
        
    def __check_set_small_sample(self) -> None:
        if self.use_tf:
            self.__check_set_small_sample_tf()
        else:
            self.__check_set_small_sample_np()
            
    def __check_set_distributions_np(self) -> None:
        seed_dist_1  = int(1e6)  # Seed for distribution 1
        seed_dist_2  = int(1e12)  # Seed for distribution 2
        if self.is_symb_1:
            if self.small_sample:
                if isinstance(self.dist_1_symb, NumpyDistribution):
                    self._dist_1_num = self.dist_1_symb.sample(self.nsamples, seed = int(seed_dist_1)).astype(self.dtype)
                elif isinstance(self._dist_1_symb, tfp.distributions.Distribution):
                    self._dist_1_num = self.dist_1_symb.sample(self.nsamples, seed = int(seed_dist_1)).numpy().astype(self.dtype) # type: ignore
                else:
                    raise ValueError("dist_1_symb should be a subclass of NumpyDistribution or tfp.distributions.Distribution.")
            else:
                self._dist_1_num = tf.convert_to_tensor([[]], dtype = self.dist_1_symb.dtype) # type: ignore
        else:
            if isinstance(self.dist_1_num, (np.ndarray, tf.Tensor)):
                self._dist_1_num = self.dist_1_num[:self.nsamples,:].astype(self.dtype)
            else:
                raise ValueError("dist_1_num should be an instance of np.ndarray or tf.Tensor.")
        if self.is_symb_2:
            if self.small_sample:
                if isinstance(self.dist_2_symb, NumpyDistribution):
                    self._dist_2_num = self.dist_2_symb.sample(self.nsamples, seed = int(seed_dist_2)).astype(self.dtype)
                elif isinstance(self.dist_2_symb, tfp.distributions.Distribution):
                    self._dist_2_num = self.dist_2_symb.sample(self.nsamples, seed = int(seed_dist_2)).numpy().astype(self.dtype) # type: ignore
                else:
                    raise ValueError("dist_2_symb should be a subclass of NumpyDistribution or tfp.distributions.Distribution.")
            else:
                self._dist_2_num = tf.convert_to_tensor([[]], dtype = self.dist_2_symb.dtype) # type: ignore
        else:
            if isinstance(self.dist_2_num, (np.ndarray, tf.Tensor)):
                self._dist_2_num = self.dist_2_num[:self.nsamples,:].astype(self.dtype)
            else:  
                raise ValueError("dist_2_num should be an instance of np.ndarray or tf.Tensor.")
            
    def __check_set_distributions_tf(self,
                                     mirror_strategy: bool = False) -> None:
        # Utility functions
        def set_dist_num_from_symb(dist: tfp.distributions.Distribution,
                                   seed: int = 0) -> tf.Tensor:
            if isinstance(dist, tfp.distributions.Distribution):
                dist_num: tf.Tensor = generate_and_clean_data(dist, self.nsamples, self.nsamples, dtype = self.dtype, seed = int(seed), mirror_strategy = mirror_strategy) # type: ignore
            else:
                raise ValueError("dist should be an instance of tfp.distributions.Distribution.")
            return dist_num
        
        def return_dist_num(dist_num: DataType) -> tf.Tensor:
            if isinstance(dist_num, tf.Tensor):
                return dist_num
            else:
                raise ValueError("dist_num should be an instance of tf.Tensor.")
        
        seed_dist_1  = int(1e6)  # Seed for distribution 1
        seed_dist_2  = int(1e12)  # Seed for distribution 2
        
        dist_1_num = tf.cond(self.is_symb_1,
                             true_fn = lambda: tf.cond(self.small_sample,
                                                       true_fn = lambda: set_dist_num_from_symb(self.dist_1_symb, seed = seed_dist_1),
                                                       false_fn = lambda: return_dist_num(self.dist_1_num)),
                             false_fn = lambda: return_dist_num(self.dist_1_num))
        dist_2_num = tf.cond(self.is_symb_2,
                             true_fn = lambda: tf.cond(self.small_sample,
                                                       true_fn = lambda: set_dist_num_from_symb(self.dist_2_symb, seed = seed_dist_2),
                                                       false_fn = lambda: return_dist_num(self.dist_2_num)),
                             false_fn = lambda: return_dist_num(self.dist_2_num))
        self._dist_1_num = tf.cast(dist_1_num, self.dtype)[:self.nsamples, :] # type: ignore
        self._dist_2_num = tf.cast(dist_2_num, self.dtype)[:self.nsamples, :] # type: ignore
        
    def __check_set_distributions(self,
                                  mirror_strategy: bool = False) -> None:
        if self.use_tf:
            self.__check_set_distributions_tf(mirror_strategy = mirror_strategy)
        else:
            self.__check_set_distributions_np()
        
    def __preprocess(self, 
                     mirror_strategy: bool = False,
                     verbose: bool = False) -> None:
        # Reset random seeds
        reset_random_seeds(seed = self.seed)
        
        # Parse input distributions
        self.__parse_input_distributions(verbose = verbose)

        # Check and set dtype
        self.__check_set_dtype()
        
        # Check and set ndims
        self.__check_set_ndims()

        # Check and set nsamples
        self.__check_set_nsamples()
        
        # Check and set small sample
        self.__check_set_small_sample()

        # Check and set distributions
        self.__check_set_distributions(mirror_strategy = mirror_strategy) 
        
    @property
    def param_dict(self) -> Dict[str, Any]:
        return {"is_symb_1": bool(self.is_symb_1),
                "is_symb_2": bool(self.is_symb_2),
                "ndims": self.ndims,
                "niter": self.niter,
                "batch_size": self.batch_size,
                "dtype": self.dtype,
                "small_sample": self.small_sample}
    
@dataclass
class TwoSampleTestResult:
    def __init__(self,
                 timestamp: str,
                 test_name: str,
                 parameters: Dict[str, Any],
                 result_value: Dict[str, Optional[DataType]]
                ) -> None:
        self.timestamp: str = timestamp
        self.test_name: str = test_name
        self.result_value: Dict[str, Optional[DataType]] = result_value
        self.__dict__.update(parameters)
    
    def result_to_dataframe(self):
        return pd.DataFrame.from_dict(self.__dict__, orient="index")

    def print_result(self,
                     print_mode: str = "full"):
        for k, v in self.__dict__.items():
            if print_mode == "full":
                print(f"{k}: {v}")
            elif print_mode == "parameters":
                if k != "result_value":
                    print(f"{k}: {v}")
            else:
                raise ValueError(f"print_mode must be either 'full' or 'parameters', but got {print_mode}")
            

class TwoSampleTestResults(object):
    def __init__(self) -> None:
        self._results: List[TwoSampleTestResult] = []
    
    @property
    def results(self) -> List[TwoSampleTestResult]:
        return self._results
        
    def append(self, item: TwoSampleTestResult) -> None:
        if isinstance(item, TwoSampleTestResult):
            self._results.append(item)
        else:
            raise ValueError('Can only add TwoSampleTestResult objects to the results.')

    def __getitem__(self, index: int) -> TwoSampleTestResult:
        return self.results[index]

    def __len__(self) -> int:
        return len(self.results)

    def __repr__(self) -> str:
        return repr(self.results)
    
    def print_results(self,
                      print_mode: str = "full"
                     ) -> None:
        for result in self.results:
            print("--------------------------------------------------------")  # add a blank line between results
            result.print_result(print_mode = print_mode)
    
    def get_results_as_dataframe(self,
                                 sort_kwargs: dict = {"by": ["batch_size","niter"], "ascending": [True]},
                                 print_mode: str = "full"
                                ) -> pd.DataFrame:
        df = pd.DataFrame()
        for result in self.results:
            df = pd.concat([df, result.result_to_dataframe().T])
        if print_mode == "full":
            df = df.sort_values(**sort_kwargs)
        elif print_mode == "parameters":
            df = df.drop(columns=["result_value"]).sort_values(**sort_kwargs)
        else:
            raise ValueError(f"print_mode must be either 'full' or 'parameters', but got {print_mode}")
        return df
    
    @property
    def results_df(self) -> pd.DataFrame:
        df = self.get_results_as_dataframe(print_mode = "full")
        return df
    
    @property
    def results_dict(self) -> Dict[str, Any]:
        return {str(k): v for k, v in self.results_df.to_dict(orient="index").items()}

    @property
    def results_params_df(self) -> pd.DataFrame:
        df = self.get_results_as_dataframe(print_mode = "parameters")
        return df
    
    @property
    def results_params_dict(self) -> Dict[str, Any]:
        return {str(k): v for k, v in self.results_params_df.to_dict(orient="index").items()}


class TwoSampleTestBase(ABC):
    """
    Base class for metrics.
    """
    def __init__(self, 
                 data_input: TwoSampleTestInputs,
                 progress_bar: bool = False,
                 verbose: bool = False
                ) -> None:
        self.Inputs: TwoSampleTestInputs = data_input
        self.progress_bar: bool = progress_bar
        self.verbose: bool = verbose
        self._start: float = 0.
        self._end: float = 0.
        self.pbar: tqdm = tqdm(disable=True)
        self._Results: TwoSampleTestResults = TwoSampleTestResults()
        
        
    @property
    def Inputs(self) -> TwoSampleTestInputs: # type: ignore
        return self._Inputs
    
    @Inputs.setter
    def Inputs(self, # type: ignore
                Inputs: TwoSampleTestInputs) -> None:
        if isinstance(Inputs, TwoSampleTestInputs):
            self._Inputs: TwoSampleTestInputs = Inputs
        else:
            raise TypeError(f"Inputs must be of type TwoSampleTestInputs, but got {type(Inputs)}")
        
    @property
    def progress_bar(self) -> bool: # type: ignore
        return self._progress_bar
    
    @progress_bar.setter
    def progress_bar(self, # type: ignore
                     progress_bar: bool) -> None:
        if isinstance(progress_bar, bool):
            self._progress_bar: bool = progress_bar
            if self.Inputs.use_tf and self.progress_bar:
                self._progress_bar = False
                print("progress_bar is disabled when using tensorflow mode.")
        else:
            raise TypeError(f"progress_bar must be of type bool, but got {type(progress_bar)}")
        
    @property
    def verbose(self) -> bool: # type: ignore
        return self._verbose

    @verbose.setter
    def verbose(self, # type: ignore
                verbose: Union[int,bool]
                ) -> None:
        if isinstance(verbose, bool):
            self._verbose: bool = verbose
        elif isinstance(verbose, int):
            self._verbose = bool(verbose)
        else:
            raise TypeError("verbose must be a bool or an int (which is automatically converted to a bool)")
        
    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def pbar(self) -> tqdm: # type: ignore
        return self._pbar
    
    @pbar.setter
    def pbar(self, # type: ignore
                pbar: tqdm) -> None:
        if isinstance(pbar, tqdm):
            self._pbar: tqdm = pbar
        else:
            raise TypeError(f"pbar must be of type tqdm, but got {type(pbar)}")
        
    @property
    def Results(self) -> TwoSampleTestResults:
        return self._Results
    
    @property
    def use_tf(self) -> bool:
        return self.Inputs.use_tf
    
    @property
    def small_sample(self) -> bool:
        return self.Inputs.small_sample
            
    def get_niter_batch_size_np(self) -> Tuple[int, int]:
        nsamples = self.Inputs.nsamples
        batch_size = self.Inputs.batch_size
        niter = self.Inputs.niter
        if nsamples < batch_size * niter:
            batch_size = nsamples // niter
        else:
            pass
        if batch_size == 0:
            raise ValueError("batch_size should be positive integer and number of samples should be larger than number of iterations.")
        return niter, batch_size
    
    def get_niter_batch_size_tf(self) -> Tuple[tf.Tensor, tf.Tensor]:
        nsamples: tf.Tensor = tf.cast(self.Inputs.nsamples, dtype = tf.int32) # type: ignore
        batch_size: tf.Tensor = tf.cast(self.Inputs.batch_size, dtype = tf.int32) # type: ignore
        niter: tf.Tensor = tf.cast(self.Inputs.niter, dtype = tf.int32) # type: ignore
        batch_size_tmp = tf.cond(nsamples < batch_size * niter,
                                    true_fn=lambda: nsamples // niter,
                                    false_fn=lambda: batch_size)
        batch_size: tf.Tensor = tf.cast(batch_size_tmp, dtype = tf.int32) # type: ignore
        tf.debugging.assert_positive(batch_size, message="batch_size should be positive integer and number of samples should be larger than number of iterations.")
        return niter, batch_size

    @property
    def param_dict(self) -> Dict[str, Any]:
        if self.Inputs.use_tf:
            niter, batch_size = self.get_niter_batch_size_np()
        else:
            niter, batch_size = self.get_niter_batch_size_tf()
        output_dict = self.Inputs.param_dict
        niter_used = niter
        output_dict["niter_used"] = int(niter_used)
        output_dict["batch_size_used"] = int(batch_size)
        output_dict["computing_time"] = self.get_computing_time()
        output_dict["small_sample"] = self.small_sample
        return output_dict    
            
    def get_computing_time(self) -> float:
        return self.end - self.start
    
    @abstractmethod
    def Test_np(self) -> None:
        pass
    
    @abstractmethod
    def Test_tf(self) -> None:
        pass