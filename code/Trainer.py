__all__ = ["Debugger",
           "HuberMinusLogProbLoss",
           "HuberMinusLogProbMetric",
           "MinusLogProbLoss",
           "MinusLogProbMetric",
           "LogProbLayer",
           "HandleNaNCallback",
           "Trainer",
           "ensure_tensor",
           "chop_to_zero",
           "chop_to_zero_array",
           "cornerplotter"
            ]
           
import os
import json
import datetime
from timeit import default_timer as timer
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
try:
    from keras.engine.keras_tensor import KerasTensor
except ImportError:
    from tensorflow.python.keras.engine.keras_tensor import KerasTensor
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from corner import corner
import Utils
from typing import Optional, Union, Tuple, List, Callable, Dict, Any, Sequence, Iterable, TypeVar, Generic, Type

DTypesInput = Union[tf.dtypes.DType,str,np.dtype,float,int]
DTypesType = Union[tf.dtypes.DType,str,np.dtype]
CustomType1 = Union[tf.Tensor,np.ndarray]
CustomType2 = Dict[str,Optional[Union[str,Dict[str,Any]]]]
CustomType3 = Dict[str,bool]
CustomType4 = Dict[str,Union[str,List[Union[str,Dict]],Dict[str,Any]]]
CustomType5 = Union[str,Dict[str,Any]]
CustomType6 = Union[str,Dict[str,Any]]
OptimizerType = Type[tf.keras.optimizers.Optimizer]
OptimizerInstanceType = (tf.keras.optimizers.Optimizer)
#OptimizerType = Type[Union[tf.keras.optimizers.Optimizer, tf.keras.optimizers.legacy.Optimizer]]
#OptimizerInstanceType = (tf.keras.optimizers.Optimizer, tf.keras.optimizers.legacy.Optimizer)
TensorType = Union[tf.Tensor,KerasTensor]
TensorInstanceType: Tuple[type,type] = (tf.Tensor,KerasTensor)

## Check TensorFlow version
#tf_major_version, tf_minor_version, _ = tf.__version__.split('.')
#is_legacy_optimizer = int(tf_major_version) < 2 or (int(tf_major_version) == 2 and int(tf_minor_version) < 10)
#if is_legacy_optimizer:
#    OptimizerType = Type[tf.keras.optimizers.Optimizer]
#    OptimizerInstanceType = tuple(tf.keras.optimizers.Optimizer)
#else:
#    OptimizerType = Type[Union[tf.keras.optimizers.Optimizer, tf.keras.optimizers.legacy.Optimizer]]
#    OptimizerInstanceType = (tf.keras.optimizers.Optimizer, tf.keras.optimizers.legacy.Optimizer)


class Debugger(object):
    """	
    A class that contains a debug mode flag.	
    """
    def __init__(self, debug_print_mode: bool = False) -> None:
        self.debug_print_mode = debug_print_mode # If True, prints debug information
        
    @property
    def debug_print_mode(self) -> bool:
        return self._debug_print_mode
    
    @debug_print_mode.setter
    def debug_print_mode(self, debug_print_mode: bool) -> None:
        if not isinstance(debug_print_mode, bool):
            raise ValueError('The debug_print_mode must be a boolean.')
        self._debug_print_mode: bool = debug_print_mode


class MinusLogProbLoss(tf.keras.losses.Loss):
    """
    Container class for -log_prob loss function
    """
    def __init__(self,
                 name: str = "MinusLogProbLoss",
                 ignore_nans: bool = False,
                 nan_threshold: float = 0.1,  # New argument
                 debug_print_mode: bool = False,  # If True, prints debug information
                 **kwargs
                ) -> None:
        super().__init__(name=name, **kwargs)
        self.ignore_nans = ignore_nans
        self.nan_threshold = nan_threshold  # Initialize new argument
        self.debug_print_mode = debug_print_mode

    def call(self, y_true, y_pred):
        loss = -y_pred
        if self.ignore_nans:
            non_nan = tf.math.is_finite(loss)
            sum_non_nan = tf.math.reduce_sum(tf.boolean_mask(loss, non_nan))
            num_non_nan = tf.math.reduce_sum(tf.cast(non_nan, tf.float32))
            total_elements = tf.math.reduce_sum(tf.ones_like(loss, dtype=tf.float32))

            # Calculate the fraction of non-NaN elements
            fraction_non_nan = num_non_nan / total_elements
            
            if fraction_non_nan < 1.0:
                # Compute mean loss based on the threshold
                if fraction_non_nan > 1.0 - self.nan_threshold:
                    tf.print(f"Warning: The fraction of NaNs in loss is below threshold. Removing them from average.")
                    mean_loss = sum_non_nan / num_non_nan
                else:
                    tf.print(f"Warning: The fraction of NaNs in loss is above threshold. Loss will be NaN.")
                    mean_loss = tf.reduce_mean(loss)
            else:
                mean_loss = tf.reduce_mean(loss)
        else:
            mean_loss = tf.reduce_mean(loss)
        
        if self.debug_print_mode:
            try:
                mean_loss = tf.debugging.check_numerics(mean_loss, "Loss calculation got NaN or Inf elements!")
            except tf.errors.InvalidArgumentError:
                tf.print("----------- MinusLogProbLoss.call() debug information -----------")
                tf.print("Final result:")
                tf.print("loss: ", mean_loss)
                raise

        return mean_loss


#class MinusLogProbLoss(tf.keras.losses.Loss, Debugger):
#    """
#    Container class for -log_prob loss function
#    """
#    def __init__(self,
#                 name: str = "MinusLogProbLoss",
#                 ignore_nans: bool = False,
#                 debug_print_mode: bool = False, # If True, prints debug information
#                 **kwargs
#                ) -> None:
#        Debugger.__init__(self, debug_print_mode = debug_print_mode) # Initialize Debugger class
#        tf.keras.losses.Loss.__init__(self, name = name, **kwargs)
#        self.ignore_nans = ignore_nans
#
#    def call(self, y_true, y_pred):
#        loss = -y_pred
#        if self.ignore_nans:
#            non_nan = tf.math.is_finite(loss)
#            sum_non_nan = tf.math.reduce_sum(tf.boolean_mask(loss, non_nan))
#            num_non_nan = tf.math.reduce_sum(tf.cast(non_nan, tf.float32))
#            mean_loss = sum_non_nan / num_non_nan
#        else:
#            mean_loss = tf.reduce_mean(loss)
#        if self.debug_print_mode:
#            try:
#                mean_loss = tf.debugging.check_numerics(mean_loss, "Loss calculation got NaN or Inf elements!")
#            except tf.errors.InvalidArgumentError:
#                tf.print("----------- MinusLogProbLoss.call() debug information -----------")
#                tf.print("Final result:")
#                tf.print("loss: ", mean_loss)
#                raise
#        return mean_loss
    
    
class MinusLogProbMetric(tf.keras.metrics.Metric):
    def __init__(self,
                 name: str = "MinusLogProbMetric",
                 ignore_nans: bool = False,
                 debug_print_mode: bool = False,
                 **kwargs
                ) -> None:
        super().__init__(name=name, **kwargs)
        self.ignore_nans = ignore_nans
        self.debug_print_mode = debug_print_mode
        self.total = self.add_weight(name="total", initializer="zeros")
        self.non_nan_count = self.add_weight(name="non_nan_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = -y_pred
        if self.ignore_nans:
            non_nan = tf.math.is_finite(loss)
            sum_non_nan = tf.math.reduce_sum(tf.boolean_mask(loss, non_nan))
            num_non_nan = tf.math.reduce_sum(tf.cast(non_nan, tf.float32))
            self.total.assign_add(sum_non_nan)
            self.non_nan_count.assign_add(num_non_nan)
        else:
            self.total.assign_add(tf.math.reduce_sum(loss))
            self.non_nan_count.assign_add(tf.cast(tf.size(loss), dtype=tf.float32))

        if self.debug_print_mode:
            try:
                result = self.total / self.non_nan_count
                result = tf.debugging.check_numerics(result, "Metric result has NaN or Inf elements!")
            except tf.errors.InvalidArgumentError:
                tf.print("----------- MinusLogProbMetric.update_state() debug information -----------")
                tf.print("Input arguments:")
                tf.print("y_true: ", y_true)
                tf.print("y_pred: ", y_pred)
                tf.print("Local variables:")
                tf.print("loss: ", loss)
                tf.print("Final result:")
                tf.print("total: ", self.total)
                tf.print("non_nan_count: ", self.non_nan_count)
                tf.print("total/non_nan_count: ", result) # type: ignore
                raise

    def result(self):
        return self.total / self.non_nan_count

    def reset_state(self):
        self.total.assign(0.0)
        self.non_nan_count.assign(0.0)


#class MinusLogProbMetric(tf.keras.metrics.Metric, Debugger):
#    def __init__(self, 
#                 name: str = "MinusLogProbMetric",
#                 debug_print_mode: bool = False, # If True, prints debug information
#                 **kwargs
#                ) -> None:
#        Debugger.__init__(self, debug_print_mode = debug_print_mode) # Initialize Debugger class
#        tf.keras.metrics.Metric.__init__(self, name = name, **kwargs)
#        self.total: tf.VariableAggregation = self.add_weight(name="total", initializer="zeros")
#        self.count: tf.VariableAggregation = self.add_weight(name="count", initializer="zeros")
#
#    def update_state(self,
#                     y_true: tf.Tensor,
#                     y_pred: tf.Tensor,
#                     sample_weight: Optional[tf.Tensor] = None
#                    ) -> None:
#        loss: tf.Tensor = -y_pred # type: ignore
#        self.total.assign_add(tf.reduce_sum(loss)) # type: ignore
#        self.count.assign_add(tf.cast(tf.size(loss), dtype=tf.float32)) # type: ignore
#        
#        if self.debug_print_mode:
#            try:
#                result: tf.Tensor = self.total / self.count # type: ignore
#                result = tf.debugging.check_numerics(result, "Metric result has NaN or Inf elements!") # type: ignore
#            except tf.errors.InvalidArgumentError:
#                tf.print("----------- HuberMinusLogProbMetric.update_state() debug information -----------")
#                tf.print("Input arguments:")
#                tf.print("y_true: ", y_true)
#                tf.print("y_pred: ", y_pred)
#                tf.print("Local variables:")
#                tf.print("loss: ", loss)
#                tf.print("Final result:")
#                tf.print("total: ", self.total)
#                tf.print("count: ", self.count)
#                tf.print("total/count: ", result) # type: ignore
#                raise
#
#    def result(self) -> tf.Tensor:
#        result: tf.Tensor = self.total / self.count # type: ignore
#        return result
#
#    def reset_state(self) -> None:
#        self.total.assign(0.) # type: ignore
#        self.count.assign(0.) # type: ignore


class MinusLogProbVarLoss(tf.keras.losses.Loss, Debugger):
    """
    Container class for -log_prob loss function
    """
    def __init__(self,
                 name: str = "MinusLogProbVarLoss",
                 debug_print_mode: bool = False, # If True, prints debug information
                 **kwargs
                ) -> None:
        Debugger.__init__(self, debug_print_mode = debug_print_mode) # Initialize Debugger class
        tf.keras.losses.Loss.__init__(self, name = name, **kwargs)

    def call(self, y_true, y_pred):
        mean_log_prob = tf.reduce_mean(y_pred)
        variance_log_prob = tf.reduce_mean((y_pred - mean_log_prob)**2)

        # Normalize mean to [0,1] range assuming it originally is in [-inf, inf] range
        normalized_mean = tf.sigmoid(mean_log_prob)

        # Normalize variance to [0,1] range assuming it originally is in [0, inf] range
        normalized_variance = tf.nn.softplus(variance_log_prob)
        loss: tf.Tensor = -(normalized_mean - normalized_variance)  # or some combination
        if self.debug_print_mode:
            try:
                loss = tf.debugging.check_numerics(loss, "Metric result has NaN or Inf elements!") # type: ignore
            except tf.errors.InvalidArgumentError:
                tf.print("----------- MinusLogProbLoss.call() debug information -----------")
                tf.print("Final result:")
                tf.print("loss: ", loss)
                raise
        return loss
    

class MinusLogProbVarMetric(tf.keras.metrics.Metric, Debugger):
    def __init__(self, 
                 name: str = "MinusLogProbVarMetric",
                 debug_print_mode: bool = False, # If True, prints debug information
                 **kwargs
                ) -> None:
        Debugger.__init__(self, debug_print_mode = debug_print_mode) # Initialize Debugger class
        tf.keras.metrics.Metric.__init__(self, name = name, **kwargs)
        self.total: tf.VariableAggregation = self.add_weight(name="total", initializer="zeros")
        self.count: tf.VariableAggregation = self.add_weight(name="count", initializer="zeros")

    def update_state(self,
                     y_true: tf.Tensor,
                     y_pred: tf.Tensor,
                     sample_weight: Optional[tf.Tensor] = None
                    ) -> None:
        mean_log_prob = tf.reduce_mean(y_pred)
        variance_log_prob = tf.reduce_mean((y_pred - mean_log_prob)**2)

        # Normalize mean to [0,1] range assuming it originally is in [-inf, inf] range
        normalized_mean = tf.sigmoid(mean_log_prob)

        # Normalize variance to [0,1] range assuming it originally is in [0, inf] range
        normalized_variance = tf.nn.softplus(variance_log_prob)
        loss: tf.Tensor = -(normalized_mean - normalized_variance)  # or some combination
        self.total.assign_add(tf.reduce_sum(loss)) # type: ignore
        self.count.assign_add(tf.cast(tf.size(loss), dtype=tf.float32)) # type: ignore
        
        if self.debug_print_mode:
            try:
                result: tf.Tensor = self.total / self.count # type: ignore
                result = tf.debugging.check_numerics(result, "Metric result has NaN or Inf elements!") # type: ignore
            except tf.errors.InvalidArgumentError:
                tf.print("----------- HuberMinusLogProbMetric.update_state() debug information -----------")
                tf.print("Input arguments:")
                tf.print("y_true: ", y_true)
                tf.print("y_pred: ", y_pred)
                tf.print("Local variables:")
                tf.print("loss: ", loss)
                tf.print("Final result:")
                tf.print("total: ", self.total)
                tf.print("count: ", self.count)
                tf.print("total/count: ", result) # type: ignore
                raise

    def result(self) -> tf.Tensor:
        result: tf.Tensor = self.total / self.count # type: ignore
        return result

    def reset_state(self) -> None:
        self.total.assign(0.) # type: ignore
        self.count.assign(0.) # type: ignore


class HuberMinusLogProbLoss(tf.keras.losses.Loss, Debugger):
    """
    Huber-like -log_prob loss function. It is a combination of Huber loss and -log_prob loss. It is defined as:
    loss = delta^2 * (sqrt(1 + (y_pred/delta)^2) - 1) - y_pred
    For large values of y_pred, the loss is approximately equal to delta^2 * (sqrt(1 + (y_pred/delta)^2) - 1), 
    while for small values of y_pred, the loss is approximately equal to y_pred^2/2.
    
    """
    def __init__(self,
                 delta: float = 5.0,
                 name: str = "HuberMinusLogProbLoss",
                 debug_print_mode: bool = False, # If True, prints debug information
                 **kwargs
                ) -> None:
        Debugger.__init__(self, debug_print_mode = debug_print_mode) # Initialize Debugger class
        self.delta = delta
        tf.keras.losses.Loss.__init__(self, name = name, **kwargs)

    @property
    def delta(self) -> float:
        return self._delta
    
    @delta.setter
    def delta(self, delta: float) -> None:
        if not isinstance(delta, float):
            raise TypeError("delta must be a float")
        self._delta: float = delta

    def call(self,
             y_true: tf.Tensor,
             y_pred: tf.Tensor
            ) -> tf.Tensor:
        delta: tf.Tensor = tf.cast(self.delta, y_pred.dtype) # type: ignore
        minus_log_prob: tf.Tensor = -y_pred  # negative log-probability # type: ignore
        abs_log_prob: tf.Tensor = tf.abs(minus_log_prob)
        small_abs_cond: tf.Tensor = abs_log_prob < delta # type: ignore
        # apply the sign of error in both types of losses
        small_abs_loss: tf.Tensor = 0.5 * minus_log_prob * abs_log_prob # type: ignore
        large_abs_loss: tf.Tensor = delta * (abs_log_prob - 0.5 * delta) * tf.sign(minus_log_prob) # type: ignore
        loss: tf.Tensor = tf.where(small_abs_cond, small_abs_loss, large_abs_loss)
        
        if self.debug_print_mode:
            try:
                loss = tf.debugging.check_numerics(loss, "Metric result has NaN or Inf elements!") # type: ignore
            except tf.errors.InvalidArgumentError:
                tf.print("----------- HuberMinusLogProbLoss.call() debug information -----------")
                tf.print("Input arguments:")
                tf.print("y_true: ", y_true)
                tf.print("y_pred: ", y_pred)
                tf.print("Local variables:")
                tf.print("minus_log_prob: ", minus_log_prob)
                tf.print("abs_log_prob: ", abs_log_prob)
                tf.print("small_abs_cond: ", small_abs_cond)
                tf.print("small_abs_loss: ", small_abs_loss)
                tf.print("large_abs_loss: ", large_abs_loss)
                tf.print("Final result:")
                tf.print("loss: ", loss)
                raise
            
        return loss


class HuberMinusLogProbMetric(tf.keras.metrics.Metric, Debugger):
    """
    Huber-like -log_prob metric function
    """
    def __init__(self,
                 delta: float = 5.0,
                 name: str = "HuberMinusLogProbMetric",
                 debug_print_mode: bool = False, # If True, prints debug information
                 **kwargs
                ) -> None:
        Debugger.__init__(self, debug_print_mode = debug_print_mode) # Initialize Debugger class
        self.delta = delta
        tf.keras.metrics.Metric.__init__(self, name = name, **kwargs)
        self.total: tf.VariableAggregation = self.add_weight(name="total", initializer="zeros")
        self.count: tf.VariableAggregation = self.add_weight(name="count", initializer="zeros")
        
    @property
    def delta(self) -> float:
        return self._delta
    
    @delta.setter
    def delta(self, delta: float) -> None:
        if not isinstance(delta, float):
            raise TypeError("delta must be a float")
        self._delta: float = delta
    
    def update_state(self,
                     y_true: tf.Tensor,
                     y_pred: tf.Tensor,
                     sample_weight: Optional[tf.Tensor] = None
                    ) -> None:
        delta: tf.Tensor = tf.cast(self.delta, y_pred.dtype) # type: ignore
        minus_log_prob: tf.Tensor = -y_pred  # negative log-probability # type: ignore
        abs_log_prob: tf.Tensor = tf.abs(minus_log_prob) # type: ignore
        small_abs_cond: tf.Tensor = abs_log_prob < delta # type: ignore
        # apply the sign of error in both types of losses
        small_abs_loss: tf.Tensor = 0.5 * minus_log_prob * abs_log_prob # type: ignore
        large_abs_loss: tf.Tensor = delta * (abs_log_prob - 0.5 * delta) * tf.sign(minus_log_prob) # type: ignore
        loss: tf.Tensor = tf.where(small_abs_cond, small_abs_loss, large_abs_loss)
                    
        self.total.assign_add(tf.reduce_sum(loss)) # type: ignore
        self.count.assign_add(tf.cast(tf.size(loss), dtype=tf.float32)) # type: ignore
            
        if self.debug_print_mode:
            try:
                result: tf.Tensor = self.total / self.count # type: ignore
                result = tf.debugging.check_numerics(result, "Metric result has NaN or Inf elements!") # type: ignore
            except tf.errors.InvalidArgumentError:
                tf.print("----------- HuberMinusLogProbMetric.update_state() debug information -----------")
                tf.print("Input arguments:")
                tf.print("y_true: ", y_true)
                tf.print("y_pred: ", y_pred)
                tf.print("Local variables:")
                tf.print("minus_log_prob: ", minus_log_prob)
                tf.print("abs_log_prob: ", abs_log_prob)
                tf.print("small_abs_cond: ", small_abs_cond)
                tf.print("small_abs_loss: ", small_abs_loss)
                tf.print("large_abs_loss: ", large_abs_loss)
                tf.print("loss: ", loss)
                tf.print("Final result:")
                tf.print("total: ", self.total)
                tf.print("count: ", self.count)
                tf.print("total/count: ", result) # type: ignore
                raise
            
    def result(self) -> tf.Tensor:
        result: tf.Tensor = self.total / self.count # type: ignore
        return result

    def reset_state(self) -> None:
        self.total.assign(0.) # type: ignore
        self.count.assign(0.) # type: ignore

    
class MinusLogProbPowerLoss(tf.keras.losses.Loss, Debugger):
    """
    Container class for -log_prob loss function
    """
    def __init__(self,
                 exp: float = 1.,
                 threshold: float = 20.,
                 name: str = "MinusLogProbPowerLoss",
                 debug_print_mode: bool = False, # If True, prints debug information
                 **kwargs
                ) -> None:
        Debugger.__init__(self, debug_print_mode = debug_print_mode) # Initialize Debugger class
        self.exp = exp
        self.threshold = threshold
        tf.keras.losses.Loss.__init__(self, name = name, **kwargs)

    def call(self, y_true, y_pred):
        linear_loss = -y_pred
        power_loss = -y_pred**self.exp
        # calculate a weight based on the difference between y_pred and y_pred_mean
        alpha = tf.exp(-tf.abs(y_pred - self.threshold))
        loss = alpha * power_loss + (1 - alpha) * linear_loss
        #condition = tf.less(-y_pred, self.threshold)
        #loss = tf.where(condition, -y_pred**self.exp, -y_pred)
        if self.debug_print_mode:
            try:
                loss = tf.debugging.check_numerics(loss, "Metric result has NaN or Inf elements!")
            except tf.errors.InvalidArgumentError:
                tf.print("----------- MinusLogProbLoss.call() debug information -----------")
                tf.print("Final result:")
                tf.print("loss: ", loss)
                raise
        return loss
    
    @property
    def exp(self) -> float:
        return self._exp
    
    @exp.setter
    def exp(self, exp: float) -> None:
        if not isinstance(exp, float):
            raise TypeError("exp must be a float")
        self._exp: float = exp
        
    @property
    def threshold(self) -> float:
        return self._threshold
    
    @threshold.setter
    def threshold(self, threshold: float) -> None:
        if not isinstance(threshold, float):
            raise TypeError("threshold must be a float")
        self._threshold: float = threshold
        
        
class MinusLogProbPowerMetric(tf.keras.metrics.Metric, Debugger):
    def __init__(self, 
                 exp: float = 1.,
                 threshold: float = 20.,
                 name: str = "MinusLogProbPowerMetric",
                 debug_print_mode: bool = False, # If True, prints debug information
                 **kwargs
                ) -> None:
        Debugger.__init__(self, debug_print_mode = debug_print_mode) # Initialize Debugger class
        self.exp = exp
        self.threshold = threshold
        tf.keras.metrics.Metric.__init__(self, name = name, **kwargs)
        self.total: tf.VariableAggregation = self.add_weight(name="total", initializer="zeros")
        self.count: tf.VariableAggregation = self.add_weight(name="count", initializer="zeros")

    @property
    def exp(self) -> float:
        return self._exp
    
    @exp.setter
    def exp(self, exp: float) -> None:
        if not isinstance(exp, float):
            raise TypeError("exp must be a float")
        self._exp: float = exp
        
    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        if not isinstance(threshold, float):
            raise TypeError("threshold must be a float")
        self._threshold: float = threshold

    def update_state(self,
                     y_true: tf.Tensor,
                     y_pred: tf.Tensor,
                     sample_weight: Optional[tf.Tensor] = None
                    ) -> None:
        linear_loss: tf.Tensor = -y_pred # type: ignore
        power_loss: tf.Tensor = -y_pred**self.exp # type: ignore
        # calculate a weight based on the difference between y_pred and y_pred_mean
        alpha = tf.exp(-tf.abs(y_pred - self.threshold)) # type: ignore
        loss = alpha * power_loss + (1 - alpha) * linear_loss
        #condition = tf.less(-y_pred, self.threshold)
        #loss = tf.where(condition, -y_pred**self.exp, -y_pred)
        self.total.assign_add(tf.reduce_sum(loss)) # type: ignore
        self.count.assign_add(tf.cast(tf.size(loss), dtype=tf.float32)) # type: ignore
        
        if self.debug_print_mode:
            try:
                result: tf.Tensor = self.total / self.count # type: ignore
                result = tf.debugging.check_numerics(result, "Metric result has NaN or Inf elements!") # type: ignore
            except tf.errors.InvalidArgumentError:
                tf.print("----------- HuberMinusLogProbMetric.update_state() debug information -----------")
                tf.print("Input arguments:")
                tf.print("y_true: ", y_true)
                tf.print("y_pred: ", y_pred)
                tf.print("Local variables:")
                tf.print("loss: ", loss)
                tf.print("Final result:")
                tf.print("total: ", self.total)
                tf.print("count: ", self.count)
                tf.print("total/count: ", result) # type: ignore
                raise

    def result(self) -> tf.Tensor:
        result: tf.Tensor = self.total / self.count # type: ignore
        return result

    def reset_state(self) -> None:
        self.total.assign(0.) # type: ignore
        self.count.assign(0.) # type: ignore


class LogProbLayer(tf.keras.layers.Layer):
    """
    Layer that returns the log_prob of a distribution
    Used to add debug assertion to the distribution.log_prob() method
    """
    def __init__(self,
                 distribution: tfp.distributions.Distribution,
                 dtype: DTypesType = "float32",
                 **kwargs):
        super(LogProbLayer, self).__init__(dtype=dtype, **kwargs)
        self._distribution: tfp.distributions.Distribution = distribution

    @property
    def distribution(self) -> tfp.distributions.Distribution:
        return self._distribution

    def call(self, 
             inputs: tf.Tensor
             ) -> tf.Tensor:
        log_prob: tf.Tensor = self.distribution.log_prob(inputs)
        #tf.debugging.assert_all_finite(log_prob, 'Log probability contains NaN or inf values.')
        return log_prob


class HandleNaNCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_path, random_seed_var, lr_reduction_factor=0.5, max_restarts=3):
        super(HandleNaNCallback, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.random_seed_var = random_seed_var
        self.lr_reduction_factor = lr_reduction_factor  # The factor by which to reduce the learning rate
        self.restarts = 0  # Count of restarts
        self.max_restarts = max_restarts

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss') # type: ignore
        if np.isnan(loss) or np.isinf(loss):
            if epoch == 0:
                print("NaN or inf found in loss at first epoch. Stopping training.")
                self.model.stop_training = True # type: ignore
                return  # Exit the function

            self.restarts += 1
            if self.restarts > self.max_restarts:
                print("Reached max number of restarts. Stopping training.")
                self.model.stop_training = True # type: ignore
            else:
                print(f"NaN or inf found in loss at epoch {epoch}, resetting model to best weights, changing random seed, and reducing learning rate.")

                # Modify random seed
                new_seed = self.random_seed_var + epoch
                Utils.reset_random_seeds(new_seed)
                
                # Reset to best weights
                self.model.load_weights(self.checkpoint_path) # type: ignore

                # Reduce learning rate
                old_lr = float(self.model.optimizer.learning_rate) # type: ignore
                new_lr = old_lr * self.lr_reduction_factor
                self.model.optimizer.learning_rate = new_lr # type: ignore
                print(f"Reduced learning rate from {old_lr} to {new_lr}")

                # Continue training
                self.model.stop_training = False # type: ignore


# Custom callback to terminate training if too many NaNs
class TerminateOnNaNFractionCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to terminate training if too many NaNs (a fraction larger than threshold on the validation data)
    """
    def __init__(self, threshold=0.1, validation_data=None):
        self.threshold = threshold  # fraction of NaNs allowed
        self.validation_data = validation_data

    def on_epoch_end(self, batch, logs=None):
        log_prob_output = self.model.predict(self.validation_data, verbose = 0) # type: ignore
        num_total = np.prod(log_prob_output.shape)
        num_nans = np.isnan(log_prob_output).sum()

        nan_fraction = num_nans / num_total
        if nan_fraction > self.threshold:
            print(f"Terminate training: NaN fraction {nan_fraction} > {self.threshold}")
            self.model.stop_training = True # type: ignore


class Trainer(Debugger):
    def __init__(self, 
                 base_distribution: tfp.distributions.Distribution, # Base distribution
                 flow: tfp.bijectors.Bijector, # Flow
                 x_data_train: Union[np.ndarray, tf.data.Dataset, tf.Tensor], # Data
                 y_data_train: Union[np.ndarray, tf.data.Dataset, tf.Tensor], # Data
                 io_kwargs: Optional[Dict[str,Any]] = None, # Dictionary of arguments for the io. Example: {"save_model": True, "save_weights": True, "save_history": True, "save_best_only": True, "save_freq": "epoch"}
                 data_kwargs: Optional[Dict[str,Any]] = None, # Dictionary of arguments for the data. Example: {"seed": 0, "buffer_size": 10000}
                 compiler_kwargs: Optional[CustomType4] = None, # Dictionary of arguments for the compiler. Example: {"optimizer": "adam", "loss": "mse", "metrics": ["mse"]}
                 callbacks_kwargs: Optional[List[CustomType6]] = None, # Dictionary of arguments for the callbacks. Example: {"EarlyStopping": {"monitor": "loss", "patience": 10}}
                 fit_kwargs: Optional[Dict[str,Any]] = None, # Dictionary of arguments for the fit method. Example: {"epochs": 100, "verbose": 1}
                 debug_print_mode: bool = False # If True, prints debug information
                 ):
        Debugger.__init__(self, debug_print_mode = debug_print_mode) # Initialize Debugger class
        # Print debug information if in debug mode
        if self.debug_print_mode:
            print("\n--------------- Debub info ---------------")
            print("Initializing Trainer with following parameters:")
            print(f"base_distribution: {base_distribution}")
            print(f"flow: {flow}")
            print(f"x_data_train shape: {x_data_train.shape}")
            print(f"y_data_train shape: {y_data_train.shape}")
            print(f"io_kwargs: {io_kwargs}")
            print(f"data_kwargs: {data_kwargs}")
            print(f"compiler_kwargs: {compiler_kwargs}")
            print(f"callbacks_kwargs: {callbacks_kwargs}")
            print(f"fit_kwargs: {fit_kwargs}")

        # Parse base distribution and flow and build nf_dist distribution
        self.base_dist = base_distribution
        self.flow = flow
        self.nf_dist = tfd.TransformedDistribution(distribution = self.base_dist,
                                                   bijector = self.flow)
        
        # Print debug information if in debug mode
        if self.debug_print_mode:
            print("\n--------------- Debub info ---------------")
            print("Defined attributes:")
            print(f"self.base_dist: {self.base_dist}")
            print(f"self.flow: {self.flow}")
            print(f"self.nf_dist: {self.nf_dist}")

        # Parse I/O arguments ans set results path (also sets results_path)
        self.io_kwargs = self._get_io_args(io_kwargs)
        
        if self.debug_print_mode:
            print(f"self.io_kwargs: {self.io_kwargs}")
            print(f"self.results_path: {self.results_path}")

        # Parse data arguments (also sets property self.seed)
        self.data_kwargs = self._get_data_args(data_kwargs)
        
        if self.debug_print_mode:
            print(f"self.data_kwargs: {self.data_kwargs}")

        # Reset random seeds
        Utils.reset_random_seeds(self.seed)
        
        # Parse data
        self.x_data = x_data_train
        self.y_data = y_data_train
        self.ndims = self.x_data.shape[1] # type: ignore
        
        # Print debug information if in debug mode
        if self.debug_print_mode:
            print(f"self.x_data: {self.x_data}")
            print(f"self.y_data: {self.y_data}")
            print(f"self.ndims: {self.ndims}")

        input: tf.keras.layers.Input = tf.keras.layers.Input(shape=(self.ndims,), dtype=tf.float32) # type: ignore
        self.log_prob = LogProbLayer(self.nf_dist)(input) # type: ignore
        #self.log_prob = self.nf_dist.log_prob(input) # type: ignore
        self.model = tf.keras.Model(input, self.log_prob)
        self.trainable_params = sum(var.numpy().size for var in self.model.trainable_weights)
        self.non_trainable_params = sum(var.numpy().size for var in self.model.non_trainable_weights)

        self.total_params = self.trainable_params + self.non_trainable_params
        print("Model defined.")
        print("Model summary: ", self.model.summary())
        
        # Print debug information if in debug mode
        if self.debug_print_mode:
            print(f"self.log_prob: {self.log_prob}")
            print(f"self.model: {self.model}")
        
        # Get compile args 
        self.compiler_kwargs = compiler_kwargs or {}
        
        # Get optimizer
        optimizer_config = self.compiler_kwargs.get("optimizer")
        if isinstance(optimizer_config, (str,dict)):
            self.optimizer_config = optimizer_config
        else:
            raise TypeError("optimizer must be a string or a dictionary.")
        print(f"self.optimizer_config: {self.optimizer_config}")
        self.optimizer = tf.keras.optimizers.get(self.optimizer_config)
        
        # Print debug information if in debug mode
        if self.debug_print_mode:
            print(f"self.optimizer_config: {self.optimizer_config}")
            print(f"self.optimizer: {self.optimizer}")
        
        # Get loss
        loss_config = self.compiler_kwargs.get("loss")
        if isinstance(loss_config, (str,dict)):
            self.loss_config = loss_config
        else:
            raise TypeError("loss must be a string or a dictionary.")
        self.loss = self._get_loss(self.loss_config)
            
        # Print debug information if in debug mode 
        if self.debug_print_mode:
            print(f"self.loss_config: {self.loss_config}")
            print(f"self.loss: {self.loss}")
            
        # Get metrics
        metrics_configs = self.compiler_kwargs.get("metrics")
        if isinstance(metrics_configs, list):
            self.metrics_configs = metrics_configs
            self.metrics = [self._get_metric(metric_config) for metric_config in self.metrics_configs if self._get_metric(metric_config) is not None]
        elif metrics_configs is None:
            self.metrics_configs = None
            self.metrics = None
        else:
            raise TypeError("metrics must be a list of strings or dictionaries or 'None'.")
        
        # Print debug information if in debug mode
        if self.debug_print_mode:
            print(f"self.metrics_configs: {self.metrics_configs}")
            print(f"self.metrics: {self.metrics}")
            
        compile_kwargs = self.compiler_kwargs.get("compile_kwargs")
        if isinstance(compile_kwargs, dict):
            self.compile_kwargs = compile_kwargs
            
        # Print debug information if in debug mode
        if self.debug_print_mode:
            print(f"self.compile_kwargs: {self.compile_kwargs}")
        
        # Get callbacks
        self.callbacks_configs = callbacks_kwargs
        if isinstance(self.callbacks_configs, list):
            self.callbacks = []
            for callback_config in self.callbacks_configs:
                tmp = self._get_callback(callback_config)
                if tmp is not None:
                    self.callbacks.append(tmp)
        elif self.callbacks_configs is None:
            self.callbacks = None
        else:
            raise TypeError("callbacks must be a list of strings or dictionaries or 'None'.")
        
        # Print debug information if in debug mode
        if self.debug_print_mode:
            print(f"self.callbacks_configs: {self.callbacks_configs}")
            print(f"self.callbacks: {self.callbacks}")
        
        # Parse fit kwargs
        self.fit_kwargs = self._get_fit_args(fit_kwargs)

        # Print debug information if in debug mode
        if self.debug_print_mode:      
            print(f"self.fit_kwargs: {self.fit_kwargs}")

        self.is_compiled = False
        self.training_time = 0.0
        self.history = {}
        
        if self.debug_print_mode:
            print(f"self.is_compiled: {self.is_compiled}")
            print(f"self.training_time: {self.training_time}")
            print(f"self.history: {self.history}")
            
        # Compile model and load weights
        self.compile()
        if self.load_weights:
            self.load_model_weights()
        if self.load_results:
            self.load_model_history()
            
    @property
    def base_dist(self) -> tfp.distributions.Distribution:
        return self._base_dist

    @base_dist.setter
    def base_dist(self, value: tfp.distributions.Distribution) -> None:
        if not isinstance(value, tfp.distributions.Distribution):
            raise ValueError("value is not of type tfp.distributions.Distribution.")
        try:
            _ = self._base_dist
            raise ValueError("base_dist can only be set at init time and can not be changed.")
        except AttributeError:
            self._base_dist: tfp.distributions.Distribution = value
            
    @property
    def flow(self) -> tfp.bijectors.Bijector:
        return self._flow
        
    @flow.setter
    def flow(self, value: tfp.bijectors.Bijector) -> None:
        if not isinstance(value, tfp.bijectors.Bijector):
            raise ValueError("value is not of type tfp.bijectors.Bijector.")
        try:
            _ = self._flow
            raise ValueError("flow can only be set at init time and can not be changed.")
        except AttributeError:
            self._flow: tfp.bijectors.Bijector = value
    
    @property
    def nf_dist(self) -> tfp.distributions.Distribution:
        return self._nf_dist
        
    @nf_dist.setter
    def nf_dist(self, value: tfp.distributions.Distribution) -> None:
        if not isinstance(value, tfp.distributions.Distribution):
            raise ValueError("value is not of type tfp.distributions.Distribution.")
        try:
            _ = self._nf_dist
            raise ValueError("nf_dist can only be set at init time and can not be changed.")
        except AttributeError:
            self._nf_dist: tfp.distributions.Distribution = value
    
    @property
    def io_kwargs(self) -> Dict[str, Any]:
        return self._io_kwargs

    @io_kwargs.setter
    def io_kwargs(self, value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise ValueError("value is not of type dict.")
        self._io_kwargs: Dict[str, Any] = value
            
    @property
    def results_path(self) -> Path:
        return self._results_path
        
    @results_path.setter
    def results_path(self, value: Path) -> None:
        if not isinstance(value, Path):
            raise ValueError("value is not of type Path.")
        self._results_path: Path = value
   
    @property
    def load_weights(self) -> bool:
        value: Optional[bool] = self.io_kwargs.get('load_weights')
        if value is None:
            raise ValueError("value is not of type bool.")
        return value
        
    @load_weights.setter
    def load_weights(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("value is not of type bool.")
        self._io_kwargs['load_weights'] = value
    
    @property
    def load_results(self) -> bool:
        value: Optional[bool] = self.io_kwargs.get('load_results')
        if value is None:
            raise ValueError("value is not of type bool.")
        return value

    @load_results.setter
    def load_results(self, value: bool) -> None:    
        if not isinstance(value, bool):
            raise ValueError("value is not of type bool.")
        self._io_kwargs['load_results'] = value
    
    @property
    def data_kwargs(self) -> Dict[str, Any]:
        return self._data_kwargs
        
    @data_kwargs.setter
    def data_kwargs(self, value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise ValueError("value is not of type dict.")
        if "seed" not in value.keys():
            raise ValueError("data_kwargs must contain a seed item.")
        self._data_kwargs: Dict[str, Any] = value
    
    @property
    def seed(self) -> int:
        value: Optional[int] = self._data_kwargs.get('seed')
        if value is None:
            raise ValueError("value is not of type int.")
        return value

    @seed.setter
    def seed(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError("value is not of type int.")
        self._data_kwargs['seed'] = value
    
    @property
    def x_data(self) -> Union[tf.Tensor, tf.data.Dataset]:
        return self._x_data

    @x_data.setter
    def x_data(self, data: Union[np.ndarray, tf.Tensor, tf.data.Dataset]) -> None:
        if isinstance(data, np.ndarray):
            self._x_data = tf.convert_to_tensor(data, dtype=data.dtype)
        elif isinstance(data, tf.data.Dataset):
            self._x_data = data
        elif isinstance(data, tf.Tensor):
            self._x_data = data
        else:
            raise ValueError("data must be of type np.ndarray, tf.data.Dataset, or tf.Tensor")
        
    @property
    def y_data(self) -> Union[tf.Tensor, tf.data.Dataset]:
        return self._y_data

    @y_data.setter
    def y_data(self, data: Union[np.ndarray, tf.Tensor, tf.data.Dataset]) -> None:
        if isinstance(data, np.ndarray):
            self._y_data = tf.convert_to_tensor(data, dtype=data.dtype)
        elif isinstance(data, tf.data.Dataset):
            self._y_data = data
        elif isinstance(data, tf.Tensor):
            self._y_data = data
        else:
            raise ValueError("data must be of type np.ndarray, tf.data.Dataset, or tf.Tensor")
    
    @property
    def ndims(self) -> int:
        return self._ndims
    
    @ndims.setter
    def ndims(self, ndims: int) -> None:
        if not isinstance(ndims, int):
            raise ValueError('The number of particles must be an integer.')
        elif ndims <= 0:
            raise ValueError('The number of particles must be strictly positive.')
        self._ndims: int = int(self.x_data.shape[1]) # type: ignore
        
    @property
    def log_prob(self) -> Union[tf.Tensor, KerasTensor]:
        return self._log_prob
        
    @log_prob.setter
    def log_prob(self, value: Union[tf.Tensor, KerasTensor]) -> None:
        if not isinstance(value, (tf.Tensor, KerasTensor)):
            raise ValueError("value is not of type tf.Tensor.")
        try:
            _ = self._log_prob
            raise ValueError("log_prob can only be set at init time and can not be changed.")
        except AttributeError:
            self._log_prob: Union[tf.Tensor, KerasTensor] = value
        
    @property
    def model(self) -> tf.keras.Model:
        return self._model
        
    @model.setter
    def model(self, value: tf.keras.Model) -> None:
        if not isinstance(value, tf.keras.Model):
            raise ValueError("value is not of type tf.keras.Model.")
        try:
            _ = self._model
            raise ValueError("model can only be set at init time and can not be changed.")
        except AttributeError:
            self._model: tf.keras.Model = value
            
    @property
    def trainable_params(self) -> int:
        return self._trainable_params

    @trainable_params.setter
    def trainable_params(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError("value is not of type int.")
        try:
            _ = self._trainable_params
            raise ValueError("trainable_params can only be set at init time and can not be changed.")
        except AttributeError:
            self._trainable_params: int = value
            
    @property
    def non_trainable_params(self) -> int:
        return self._non_trainable_params

    @non_trainable_params.setter
    def non_trainable_params(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError("value is not of type int.")
        try:
            _ = self._non_trainable_params
            raise ValueError("non_trainable_params can only be set at init time and can not be changed.")
        except AttributeError:
            self._non_trainable_params: int = value
            
    @property
    def total_params(self) -> int:
        return self._total_params

    @total_params.setter
    def total_params(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError("value is not of type int.")
        try:
            _ = self._total_params
            raise ValueError("total_params can only be set at init time and can not be changed.")
        except AttributeError:
            self._total_params: int = value
        
    @property
    def compiler_kwargs(self) -> CustomType4:
        return self._compiler_kwargs
    
    @compiler_kwargs.setter
    def compiler_kwargs(self, compiler_kwargs: CustomType4) -> None:
        compiler_kwargs_default: CustomType4 = {'optimizer': {'class_name': 'Adam', 'config': {'learning_rate': 0.001}},
                                                'loss': {'class_name': 'HuberMinusLogProbLoss', 'config': {'delta': 5.0}},
                                                'metrics': [],#[{'class_name': 'HuberMinusLogProbMetric', 'config': {'delta': 5.0}},
                                                                 # {'class_name': 'MinusLogProbMetric', 'config': {}}],
                                                'compile_kwargs': {}}
        types_valid: Tuple[type,type,type] = (str, dict, list)
        if not isinstance(compiler_kwargs, dict):
            raise TypeError("The compiler_kwargs parameter must be a dictionary with keys 'optimizer', 'loss', 'metrics' and 'compile_kwargs'.")
        elif isinstance(compiler_kwargs, dict):
            for key, val in compiler_kwargs.items():
                if not isinstance(key, str):
                    raise TypeError("The activation parameter must be a dictionary with string keys.")
                if not isinstance(val, types_valid):
                    raise TypeError("The activation parameter must be a dictionary with string or dictionary values.")
                if isinstance(val, dict):
                    for key2 in val.keys():
                        if not isinstance(key2, str):
                            raise TypeError("The activation parameter must be a dictionary with string keys.")
                elif isinstance(val, list):
                    for item in val:
                        if not isinstance(item, (str,dict)):
                            raise TypeError("The metrics list should contain strings or dictionaries.")
                        if isinstance(item, dict):
                            for key2 in item.keys():
                                if not isinstance(key2, str):
                                    raise TypeError("The activation parameter must be a dictionary with string keys.")
        self._compiler_kwargs: CustomType4 = compiler_kwargs
        for key, default in compiler_kwargs_default.items():
            if self._compiler_kwargs.get(key) is None:
                self._compiler_kwargs[key] = default
            
    @property
    def optimizer_config(self) -> CustomType5:
        return self._optimizer_config
        
    @optimizer_config.setter
    def optimizer_config(self, optimizer_config: CustomType5) -> None:
        optimizer_config_default: Dict[str,Any] = {'class_name': 'Adam', 'config': {'learning_rate': 0.001}}
        types_valid: Tuple[type,type] = (str, dict)
        if not isinstance(optimizer_config, types_valid):
            raise TypeError("The optimizer_config parameter must be a dictionary with string keys.")
        elif isinstance(optimizer_config, dict):
            for key, val in optimizer_config.items():
                if not isinstance(key, str):
                    raise TypeError("The optimizer_config parameter must be a dictionary with string keys.")
                if not isinstance(val, types_valid):
                    raise TypeError("The optimizer_config parameter must be a dictionary with string or dictionary values.")
                if isinstance(val, dict):
                    for key2 in val.keys():
                        if not isinstance(key2, str):
                            raise TypeError("The optimizer_config parameter must be a dictionary with string keys.")
        self._optimizer_config: CustomType5 = optimizer_config
        if isinstance(self._optimizer_config, dict):
            for key, default in optimizer_config_default.items():
                if self._optimizer_config.get(key) is None:
                    self._optimizer_config[key] = default        

    @property
    def optimizer(self) -> OptimizerType:
        return self._optimizer

    @optimizer.setter   
    def optimizer(self, optimizer: OptimizerType) -> None:
        print(f"optimizer: {optimizer}")
        print(f"type(optimizer): {type(optimizer)}")
        if not isinstance(optimizer, OptimizerInstanceType):
            raise ValueError("optimizer is not of type tf.keras.optimizers.Optimizer or tf.keras.optimizers.legacy.Optimizer.")
        try:
            _ = self._optimizer
            raise ValueError("optimizer can only be set at init time and can not be changed.")
        except AttributeError:
            self._optimizer: OptimizerType = optimizer

    @property
    def loss_config(self) -> CustomType5:
        return self._loss_config
    
    @loss_config.setter
    def loss_config(self, loss_config: CustomType5) -> None:
        loss_config_default: Dict[str,Any] = {'class_name': 'MinusLogProbLoss', 'config': {}}
        types_valid: Tuple[type,type] = (str, dict)
        if not isinstance(loss_config, types_valid):
            raise TypeError("The loss_config parameter must be a dictionary with string keys or a string.")
        elif isinstance(loss_config, dict):
            for key, val in loss_config.items():
                if not isinstance(key, str):
                    raise TypeError("The loss_config parameter must be a dictionary with string keys.")
                if not isinstance(val, types_valid):
                    raise TypeError("The loss_config parameter must be a dictionary with string or dictionary values.")
                if isinstance(val, dict):
                    for key2 in val.keys():
                        if not isinstance(key2, str):
                            raise TypeError("The loss_config parameter must be a dictionary with string keys.")
        self._loss_config: CustomType5 = loss_config
        if isinstance(self._loss_config, dict):
            for key, default in loss_config_default.items():
                if self._loss_config.get(key) is None:
                    self._loss_config[key] = default
                    
    @property
    def loss(self) -> tf.keras.losses.Loss:
        return self._loss
        
    @loss.setter
    def loss(self, loss: tf.keras.losses.Loss) -> None:
        if not isinstance(loss, tf.keras.losses.Loss):
            raise ValueError("loss is not of type tf.keras.losses.Loss.")
        try:
            _ = self._loss
            raise ValueError("loss can only be set at init time and can not be changed.")
        except AttributeError:
            self._loss: tf.keras.losses.Loss = loss
        
    @property
    def metrics_configs(self) -> Optional[List[CustomType5]]:
        return self._metrics_configs
    
    @metrics_configs.setter
    def metrics_configs(self, metrics_configs: Optional[List[CustomType5]]) -> None:
        metrics_configs_default: List[CustomType5] = [{'class_name': 'MinusLogProbMetric', 'config': {}}]
        types_valid: Tuple[type,type] = (str, dict)
        if not isinstance(metrics_configs, list) and metrics_configs is not None:
            raise TypeError("The metrics_configs parameter must be a list of dictionaries.")
        elif isinstance(metrics_configs, list):
            for item in metrics_configs:
                if not isinstance(item, types_valid):
                    raise TypeError("The metrics_configs list should contain dictionaries.")
                elif isinstance(item, dict):
                    for key, val in item.items():
                        if not isinstance(key, str):
                            raise TypeError("The metrics_configs parameter must be a dictionary with string keys.")
                        if not isinstance(val, types_valid):
                            raise TypeError("The metrics_configs parameter must be a dictionary with string or dictionary values.")
                        if isinstance(val, dict):
                            for key2 in val.keys():
                                if not isinstance(key2, str):
                                    raise TypeError("The loss_config parameter must be a dictionary with string keys.")
        self._metrics_configs: Optional[List[CustomType5]] = metrics_configs
        if self._metrics_configs is not None:
            for default in metrics_configs_default:
                if isinstance(default, dict):
                    if default.get('class_name') not in [item.get('class_name') for item in self._metrics_configs if isinstance(item, dict)]:
                        self._metrics_configs.append(default)
                    
    @property
    def metrics(self) -> Optional[List[tf.keras.metrics.Metric]]:
        return self._metrics
        
    @metrics.setter
    def metrics(self, metrics: Optional[List[tf.keras.metrics.Metric]]) -> None:
        if not isinstance(metrics, list) and metrics is not None:
            raise ValueError("metrics is not of type list not 'None'.")
        elif isinstance(metrics, list):
            if not all([isinstance(metric, tf.keras.metrics.Metric) for metric in metrics]):
                raise ValueError("metrics is not of type tf.keras.metrics.Metric.")
        self._metrics: Optional[List[tf.keras.metrics.Metric]] = metrics
                    
    @property
    def compile_kwargs(self) -> Dict[str,Any]:
        return self._compile_kwargs
    
    @compile_kwargs.setter
    def compile_kwargs(self, compile_kwargs: Optional[Dict[str,Any]]) -> None:
        compile_kwargs_default: Dict[str,Any] = {}
        if not isinstance(compile_kwargs, dict) and compile_kwargs is not None:
            raise TypeError("The compile_kwargs parameter must be a dictionary or None.")
        elif isinstance(compile_kwargs, dict):
            for key, val in compile_kwargs.items():
                if not isinstance(key, str):
                    raise TypeError("The compile_kwargs parameter must be a dictionary with string keys.")
                if isinstance(val, dict):
                    for key2 in val.keys():
                        if not isinstance(key2, str):
                            raise TypeError("The compile_kwargs parameter must be a dictionary with string keys.")
        self._compile_kwargs: Dict[str,Any] = compile_kwargs or {}
        if isinstance(self._compile_kwargs, dict):
            for key, default in compile_kwargs_default.items():
                if self._compile_kwargs.get(key) is None:
                    self._compile_kwargs[key] = default
            
    @property
    def callbacks_configs(self) -> Optional[List[CustomType6]]:
        return self._callbacks_configs
    
    @callbacks_configs.setter
    def callbacks_configs(self, callbacks_configs: Optional[List[CustomType6]]) -> None:
        if callbacks_configs is None:
            self._callbacks_configs: Optional[List[CustomType6]] = None
            return
        if not isinstance(callbacks_configs, list):
            raise TypeError("The callbacks_configs parameter must be a list of dictionaries.")
        callbacks_configs_default: List[Dict[str, Any]] = []
        for item in callbacks_configs:
            if not isinstance(item, (str, dict)):
                raise TypeError("The callbacks_configs list should contain stings or dictionaries.")
            elif isinstance(item, dict):
                if not check_config_dict(item):
                    raise ValueError("The callbacks_configs list should contain valid dictionaries.")
                item = update_file_paths(item)
        self._callbacks_configs = callbacks_configs
        if self._callbacks_configs is not None:
            for default in callbacks_configs_default:
                if isinstance(default, dict):
                    if default.get('class_name') not in [item.get('class_name') for item in self._callbacks_configs if isinstance(item, dict)]:
                        self._callbacks_configs.append(default)
                        
    @property
    def callbacks(self) -> Optional[List[tf.keras.callbacks.Callback]]:
        return self._callbacks
        
    @callbacks.setter
    def callbacks(self, callbacks: Optional[List[tf.keras.callbacks.Callback]]) -> None:
        if not isinstance(callbacks, list) and callbacks is not None:
            raise ValueError("callbacks is not of type list.")
        elif isinstance(callbacks, list):
            if not all([isinstance(callback, tf.keras.callbacks.Callback) for callback in callbacks]):
                raise ValueError("callbacks is not of type tf.keras.callbacks.Callback.")
        self._callbacks: Optional[List[tf.keras.callbacks.Callback]] = callbacks

    @property
    def n_epochs(self) -> int:
        value: Optional[int] = self._fit_kwargs.get('epochs')
        if value is None:
            raise ValueError("n_epochs is not set.")
        return value
        
    @n_epochs.setter
    def n_epochs(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError("value is not of type int.")
        else:
            self._fit_kwargs['epochs'] = value
            
    @property
    def batch_size(self) -> int:
        value: Optional[int] = self._fit_kwargs.get('batch_size')
        if value is None:
            raise ValueError("batch_size is not set.")
        return value
        
    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError("value is not of type int.")
        else:
            self._fit_kwargs['batch_size'] = value
    
    @property
    def fit_kwargs(self) -> Dict[str, Any]:
        return self._fit_kwargs
        
    @fit_kwargs.setter
    def fit_kwargs(self, value: Dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise ValueError("value is not of type dict.")
        else:
            self._fit_kwargs: Dict[str, Any] = value
            
    @property
    def is_compiled(self) -> bool:
        return self._is_compiled
            
    @is_compiled.setter
    def is_compiled(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("value is not of type bool.")
        else:
            self._is_compiled: bool = value
            
    @property
    def training_time(self) -> float:
        return self._training_time
    
    @training_time.setter
    def training_time(self, value: float) -> None:
        if not isinstance(value, float):
            raise ValueError("value is not of type float.")
        else:
            self._training_time: float = value
            
    @property
    def history(self) -> Dict[str, List[float]]:
        return self._history
        
    @history.setter
    def history(self, value: Dict[str, List[float]]) -> None:
        if not isinstance(value, dict):
            raise ValueError("value is not of type dict.")
        else:
            self._history: Dict[str, List[float]] = value
        
    def _get_data_args(self,
                       data_kwargs_input: Optional[Dict[str, Any]] = None
                      ) -> Dict[str, Any]:
        """
        Get data args from their configs
        """
        # Default data args
        data_kwargs_default: Dict[str, Any] = {'seed': 0}
        
        data_kwargs: Dict[str, Any] = data_kwargs_input or {}
        
        for key, value in data_kwargs_default.items():
            if data_kwargs.get(key) is None:
                data_kwargs[key] = value
        
        return data_kwargs
            
    def _get_loss(self, 
                  loss_config: CustomType5
                 ) -> tf.keras.losses.Loss:
        if isinstance(loss_config, dict):
            class_name: Optional[str] = loss_config.get('class_name')
            if class_name is None:
                raise ValueError("loss_config must contain a class_name.")
            class_kwargs: Dict[str, Any] = loss_config.get('config', {})
            if class_name.lower() == 'huberminuslogprobloss':
                loss = HuberMinusLogProbLoss(**class_kwargs)
            elif class_name.lower() == 'minuslogprobloss':
                loss = MinusLogProbLoss(**class_kwargs)
            elif class_name.lower() == 'minuslogprobpowerloss':
                loss = MinusLogProbPowerLoss(**class_kwargs)
            else:
                try:
                    loss = tf.keras.losses.get(loss_config)
                except:
                    raise ValueError(f"Unsupported loss: {class_name}")
        elif isinstance(loss_config, str):
            class_name = loss_config
            if class_name.lower() == 'huberminuslogprobloss':
                loss = HuberMinusLogProbLoss(delta = 5.0)
            elif class_name.lower() == 'minuslogprobloss':
                loss = MinusLogProbLoss()
            elif class_name.lower() == 'minuslogprobsquaredloss':
                loss = MinusLogProbPowerLoss()
            else:
                try:
                    loss = tf.keras.losses.get(loss_config)
                except:
                    raise ValueError(f"Unsupported loss: {class_name}")
        else:
            raise ValueError(f"Unsupported loss: {loss_config}")
        return loss # type: ignore

    def _get_metric(self, 
                    metric_config: CustomType5
                   ) -> tf.keras.metrics.Metric:
        if isinstance(metric_config, dict):
            class_name: Optional[str] = metric_config.get('class_name')
            if class_name is None:
                raise ValueError("metric_config does not have a class_name.")
            class_kwargs: Dict[str, Any] = metric_config.get('config', {})
            if class_name.lower() == 'huberminuslogprobmetric':
                metric = HuberMinusLogProbMetric(**class_kwargs)
            elif class_name.lower() == 'minuslogprobmetric':
                metric = MinusLogProbMetric(**class_kwargs)
            elif class_name.lower() == 'minuslogprobpowermetric':
                metric = MinusLogProbPowerMetric(**class_kwargs)
            else:
                try:
                    metric = tf.keras.metrics.get(metric_config)
                except ValueError:
                    print(class_name, "is not a valid metric type. Skipping it.")
                    metric = None
        elif isinstance(metric_config, str):
            class_name = metric_config
            if class_name.lower() == 'huberminuslogprobmetric':
                metric = HuberMinusLogProbMetric()
            elif class_name.lower() == 'minuslogprobmetric':
                metric = MinusLogProbMetric()
            elif class_name.lower() == 'minuslogprobpowermetric':
                metric = MinusLogProbPowerMetric()
            else:
                try:
                    metric = tf.keras.metrics.get(metric_config)
                except:
                    raise ValueError(f"Unsupported metric: {class_name}")
        else:
            raise ValueError(f"Unsupported metric: {metric_config}")
        return metric
    
    def _get_callback(self,
                      callback_config: CustomType6
                     ) -> Optional[tf.keras.callbacks.Callback]:
        """
        Initialize callbacks from their configs
        """
        def get_class_from_string(class_name: str) -> Optional[tf.keras.callbacks.Callback]:
            try:
                callback_string: str = f"tf.keras.callbacks.{class_name}"
                class_obj: Optional[tf.keras.callbacks.Callback] = eval(callback_string)
            except AttributeError:
                print(class_name, "is not a valid callback type. Skipping it.")
                class_obj = None
            return class_obj
        if isinstance(callback_config, dict):
            class_name: Optional[str] = callback_config.get("class_name")
            if class_name is None:
                raise ValueError("callback_config does not have a class_name.")
            class_kwargs: Dict[str, Any] = callback_config.get("config", {})
            if class_name.lower() == 'printepochinfo':
                callback: Optional[tf.keras.callbacks.Callback] = tf.keras.callbacks.LambdaCallback(on_epoch_end=self._epoch_callback)
            elif class_name.lower() == 'handlenancallback':
                callback = HandleNaNCallback(**class_kwargs)
            elif class_name.lower() == 'terminateonnanfractioncallback':
                callback = TerminateOnNaNFractionCallback(**class_kwargs)
            else:
                class_obj = get_class_from_string(class_name)
                if class_obj is not None:
                    callback = class_obj(**class_kwargs) # type: ignore
                else:
                    callback = None
        elif isinstance(callback_config, str):
            class_name = callback_config
            if class_name.lower() == 'printepochinfo':
                callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=self._epoch_callback)
            else:
                class_obj = get_class_from_string(class_name)
                if class_obj is not None:
                    callback = class_obj() # type: ignore
                else:
                    callback = None
        else:
            raise ValueError(f"Unsupported callback: {callback_config}")
        return callback
            
    def _epoch_callback(self,
                        epoch: int,
                        logs: Dict[str, float]
                       ) -> None:
        n_disp: int = 1  # or whatever number you want to use
        if epoch % n_disp == 0:
            timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(timestamp,'\nEpoch {}/{}'.format(epoch + 1, self.n_epochs),
                  '\n\t ' + (': {:.4f}, '.join(logs.keys()) + ': {:.4f}').format(*logs.values()))

    def _get_fit_args(self,
                      fit_kwargs_input: Optional[Dict[str, Any]] = None
                      ) -> Dict[str, Any]:
        """
        Get fit args from their configs
        """
        fit_kwargs_default: Dict[str, Any] = {'batch_size': 512, 'verbose': 2}
        
        fit_kwargs: Dict[str, Any] = fit_kwargs_input or {}
        
        for key, value in fit_kwargs_default.items():
            if fit_kwargs.get(key) is None:
                fit_kwargs[key] = value
        
        return fit_kwargs
    
    def _get_io_args(self,
                     io_kwargs_input: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]: 
        """
        Get io args from their configs
        """
        io_kwargs_default: Dict[str, Any] = {'results_path': Path('results'),
                                             'load_weights': False,
                                             'load_results': False}
        
        io_kwargs = io_kwargs_input or {}
        
        for key, value in io_kwargs_default.items():
            if io_kwargs.get(key) is None:
                io_kwargs[key] = value

        # Create results directory if it doesn't exist
        self.results_path = Path(io_kwargs["results_path"])
        self.results_path.mkdir(parents=True, exist_ok=True)

        return io_kwargs

    def load_model_weights(self) -> None:
        weights_folder: Path = self.results_path.joinpath('weights')
        weights_file: Path = weights_folder.joinpath('best_weights.h5')
        try:
            self.model.load_weights(weights_file)
            print('Found and loaded existing weights.')
        except:
            print(f'No weights found in {weights_file}. Training from scratch.')
    
    def load_model_history(self) -> None:
        details_path: Path = self.results_path.joinpath('results.json')
        try:
            with details_path.open('r') as f:
                details_json: Dict[str, Any] = json.load(f)
                self._history['loss'] = details_json['train_loss_history']
                self._history['val_loss'] = details_json['val_loss_history']
                self._history['lr'] = details_json['lr_history']
                self._training_time = details_json['training_time']
                print('Found and loaded existing history.')
        except:
            print('No history found. Generating new history.')

    def compile(self, 
                force: bool = False
               ) -> None:
        """
        Compile the model
        """
        if self.is_compiled and not force:
            print('Model already compiled. Use force=True to recompile.')
            compile = False
        elif self.is_compiled and force:
            print('Recompiling model.')
            compile = True
        else:
            compile = True
        if compile:
            try:
                self.model.compile(optimizer=self.optimizer,
                                   loss=self.loss,
                                   metrics=self.metrics,
                                   **self.compile_kwargs)
                self._is_compiled = True
                print('Model successfully compiled.')
            except:
                print('Error compiling model. Check your optimizer and loss function.')
                self._is_compiled = False

    def train(self) -> None:#Tuple[Dict[str, Any], float]:
        start: float = timer()
        history: tf.keras.callbacks.History = self.model.fit(x=self.x_data,
                                                             y=self.y_data,
                                                             callbacks=self.callbacks,
                                                             **self.fit_kwargs
                                                             )
        end: float = timer()
        self._training_time += end - start
        self._history['loss'] = self._history.get("loss",[]) + history.history['loss']
        self._history['val_loss'] = self._history.get("val_loss",[]) + history.history['val_loss']
        self._history['lr'] = self._history.get("lr",[]) + history.history['lr']
        #return self._history, self._training_time

##########################################################################
####################### Standalone utility functions #####################
##########################################################################

def ensure_tensor(input: Union[TensorType, np.ndarray, List[float], float, List[int], int],
                  dtype: DTypesType) -> tf.Tensor:
    if not isinstance(input, tf.Tensor):
        return tf.cast(input, dtype=dtype) # type: ignore
    elif input.dtype != dtype:
        return tf.cast(input, dtype=dtype) # type: ignore
    else:
        return input
    
def check_config_dict(d):
    if not isinstance(d, dict):
        return False
    keys = set(d.keys())
    valid_keys = {"class_name", "config"}
    if "class_name" in keys:
        if keys != valid_keys:
            return False
        if not isinstance(d["class_name"], str):
            return False
        if not isinstance(d["config"], dict):
            return False
        return check_config_dict(d["config"])
    return True

def update_file_paths(d):
    for key, val in d.items():
        if key in ["filepath", "log_dir", "filename", "path", "file"] and isinstance(val, str):
            d[key] = os.path.abspath(val)
        if key == "config" and isinstance(val, dict):
            update_file_paths(val)
    return d

def chop_to_zero(number: Union[float, tf.Tensor, np.ndarray],
                 threshold: float = 1e-06) -> Union[float, tf.Tensor, np.ndarray]:
    """
    Chop a number to zero if it is below a threshold
    """
    if isinstance(number, tf.Tensor):
        if abs(number) < threshold: # type: ignore
            result: Union[float, tf.Tensor, np.ndarray] = tf.constant(0.0, dtype=number.dtype)
        else:
            result = number
    elif isinstance(number, np.ndarray):
        if abs(number) < threshold:
            result = np.array(0.0, dtype=number.dtype)
        else:
            result = number
    elif isinstance(number, float):
        if abs(number) < threshold:
            result = 0.0
        else:
            result = number    
    return result

def chop_to_zero_array(x: Union[tf.Tensor, np.ndarray, list, tuple],
                       threshold: float = 1e-06) -> Union[tf.Tensor, np.ndarray, list, tuple]:
    if isinstance(x, list):
        result: Union[tf.Tensor, np.ndarray, list, tuple] = [chop_to_zero_array(y, threshold) for y in x]
    elif isinstance(x, tuple):
        result = tuple([chop_to_zero_array(y, threshold) for y in x])
    elif isinstance(x, np.ndarray):
        result = np.where(np.abs(x) < threshold, 0.0, x)
    elif isinstance(x, tf.Tensor):
        result = tf.where(tf.abs(x) < threshold, 0.0, x)
    else:   
        raise TypeError("Input must be a list, tuple, NumPy array or TensorFlow tensor.")
    return result

def cornerplotter(test_data_input: Union[np.ndarray, tf.Tensor],
                  nf_data_input: Union[np.ndarray, tf.Tensor],
                  max_dim: int = 32,
                  n_bins: int = 50,
                  path_to_save_input: Optional[Union[str, Path]] = None,
                  file_name_input: Optional[str] = None,
                 ) -> None:
    # Format the input tensors
    if isinstance(test_data_input, tf.Tensor):
        test_data: np.ndarray = test_data_input.numpy() # type: ignore
    elif isinstance(test_data_input, np.ndarray):
        test_data = test_data_input
    else:
        raise TypeError("test_data_input must be a tf.Tensor or np.ndarray")
    if isinstance(nf_data_input, tf.Tensor):
        nf_data: np.ndarray = nf_data_input.numpy() # type: ignore
    elif isinstance(nf_data_input, np.ndarray):
        nf_data = nf_data_input
    else:
        raise TypeError("nf_data_input must be a tf.Tensor or np.ndarray")
    # Check that the two samples have the same number of dimensions
    if test_data.shape[1] != nf_data.shape[1]:
        raise ValueError("The two samples must have the same number of dimensions")
    else:
        ndims: int = test_data.shape[1]
    # Check/remove nans
    test_data_no_nans: np.ndarray = test_data[~np.isnan(test_data).any(axis=1), :]
    if len(test_data) != len(test_data_no_nans):
        print("Samples containing nan have been removed from test set. The fraction of nans over the total samples was:", str((len(test_data)-len(test_data_no_nans))/len(test_data)),".")
    nf_data_no_nans: np.ndarray = nf_data[~np.isnan(nf_data).any(axis=1), :]
    if len(nf_data) != len(nf_data_no_nans):
        print("Samples containing nan have been removed from nf set. The fraction of nans over the total samples was:", str((len(nf_data)-len(nf_data_no_nans))/len(nf_data)),".")
    # Check that the two samples have the same number of events
    n_test: int = test_data_no_nans.shape[0]
    n_nf: int = nf_data_no_nans.shape[0]
    if n_test != n_nf:
        print("The two samples (after removing possible nans) have different number of points. The smaller number of points will be used.")
        if n_test < n_nf:
            nf_data_no_nans = nf_data_no_nans[:n_test]
        else:
            test_data_no_nans = test_data_no_nans[:n_nf]
    test_samples: np.ndarray = test_data_no_nans
    nf_samples: np.ndarray = nf_data_no_nans            
    shape: Tuple[int, ...] = test_samples.shape
    path_to_save: Path = Path(path_to_save_input) if path_to_save_input is not None else Path("./")
    file_name: str = file_name_input.split(".")[0]+".png" if file_name_input is not None else "corner_plot.png"
    # Define generic labels
    labels: List[str] = []
    for i in range(shape[1]):
        labels.append(r"$\theta_{%d}$" % i)
        i = i+1
    # Choose dimensions to plot
    thin: int = int(shape[1]/max_dim)+1
    if thin<=2:
        thin = 1
    # Select samples to plot
    test_samples = test_samples[:,::thin]
    nf_samples = nf_samples[:,::thin]
    # Select labels
    labels = list(np.array(labels)[::thin])
    red_bins=n_bins
    density=(np.max(test_samples,axis=0)-np.min(test_samples,axis=0))/red_bins
    #
    blue_bins=(np.max(nf_samples,axis=0)-np.min(nf_samples,axis=0))/density
    blue_bins=blue_bins.astype(int).tolist()
    blue_line = mlines.Line2D([], [], color='red', label='target')
    red_line = mlines.Line2D([], [], color='blue', label='NF')
    figure=corner(test_samples,color='red',bins=red_bins,labels=[r"%s" % s for s in labels],normalize1d=True)
    corner(nf_samples,color='blue',bins=blue_bins,fig=figure,normalize1d=True)
    plt.legend(handles=[blue_line,red_line], bbox_to_anchor=(-ndims+1.8, ndims+.3, 1., 0.) ,fontsize='xx-large')
    if file_name.split(".")[1] == "png":
        plt.savefig(path_to_save.joinpath(file_name+'.png'),bbox_inches='tight',dpi=300)
    elif file_name.split(".")[1] == "pdf":
        plt.savefig(path_to_save.joinpath(file_name+'.pdf'),bbox_inches='tight',pil_kwargs={'quality':50})
    else:
        try:
            plt.savefig(path_to_save.joinpath(file_name+'.png'),bbox_inches='tight')
        except:
            raise ValueError("file_name must have a supported extension")
    plt.show()
    plt.close()
    
def train_plotter(t_losses,v_losses,path_to_plots):
    plt.plot(t_losses,label='train')
    plt.plot(v_losses,label='validation')
    plt.legend()
    plt.title('history')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(path_to_plots+'/loss_plot.pdf')
    plt.close()
    return