from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb= tfp.bijectors
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.stats import epps_singleton_2samp
from scipy.stats import anderson_ksamp
from timeit import default_timer as timer
import math
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization,Add,Dense,ReLU,Layer, Conv2D, Reshape,Flatten,Lambda
from tensorflow.keras import Model, Input,Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LambdaCallback
from statistics import mean,median
from tensorflow.python.framework.ops import disable_eager_execution
from RQS import RationalQuadraticSpline

class NN(Layer):
    """
    Neural Network Architecture for calcualting the parameters of the Cspline bijector:
        - bin widths
        - bin heights
        - knot slopes
    The call method takes the input x and passes it through a series of dense layers, 
    returning the output of each layer as a list.
    The output of this NN is passed to the Cspline bijector to calculate the transformation.
    """
    def __init__(self, 
                 tran_dims, 
                 spline_knots,
                 range_min, 
                 n_hidden = [5,5],
                 activation = "relu",
                 use_bias = True,
                 kernel_initializer = 'glorot_uniform',
                 bias_initializer = 'zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 activity_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 use_batch_norm = False):
        super(NN, self).__init__()
        
        # Set attributes
        self.tran_dims = tran_dims
        self.range_min = range_min
        self.spline_knots = spline_knots
        self.use_batch_norm = use_batch_norm
        
        # Define layers
        layer_list = []
        for i, hidden in enumerate(n_hidden):
            if self.use_batch_norm:
                layer_list.append(Dense(hidden, 
                                        #activation = activation,
                                        use_bias = use_bias,
                                        kernel_initializer = kernel_initializer,
                                        bias_initializer = bias_initializer,
                                        kernel_regularizer = kernel_regularizer,
                                        bias_regularizer = bias_regularizer,
                                        activity_regularizer = activity_regularizer,
                                        kernel_constraint = kernel_constraint,
                                        bias_constraint = bias_constraint))
                layer_list.append(BatchNormalization())
                layer_list.append(tf.keras.layers.Activation(activation))
            else:
                layer_list.append(Dense(hidden, 
                                        activation = activation,
                                        use_bias = use_bias,
                                        kernel_initializer = kernel_initializer,
                                        bias_initializer = bias_initializer,
                                        kernel_regularizer = kernel_regularizer,
                                        bias_regularizer = bias_regularizer,
                                        activity_regularizer = activity_regularizer,
                                        kernel_constraint = kernel_constraint,
                                        bias_constraint = bias_constraint))
        self.layer_list = layer_list
        
        # These are the definitions used for the Real-NVP bijector
        # self.log_s_layer = Dense(input_shape, activation="tanh", name='log_s')
        # self.t_layer = Dense(input_shape, name='t')
        
        # Define output layers for Cspline bijector
        bin_widths = []
        bin_heights = []
        knot_slopes = []
      
        # Define Dense layers for bin widths and heights
        for _ in range(spline_knots):
            bin_widths.append(Dense(tran_dims,activation=activation))
            bin_heights.append(Dense(tran_dims,activation=activation))
    
        # Define Dense layers for knot slopes
        for _ in range(spline_knots-1):
            knot_slopes.append(Dense(tran_dims,activation=activation))
            
        
        # bin_widths=Dense(self.tran_dims*spline_knots,activation="softmax")
        # bin_heights=Dense(self.tran_dims*spline_knots,activation="softmax")
        # knot_slopes=Dense(self.tran_dims*(spline_knots-1),activation="softplus")        
        
        # Set attributes for Cspline bijector
        self.bin_widths = bin_widths
        self.bin_heights = bin_heights
        self.knot_slopes = knot_slopes
        
    def call(self, x):
        # Define initial input
        y = x

        # Pass through hidden layers
        for layer in self.layer_list:
            y= layer(y)
        
        # Define output layers for Cspline bijector
        bin_widths = []
        bin_heights = []
        knot_slopes = []
        
        # Define outputs for bin widths and heights
        for j in range(self.spline_knots):
            bin_widths.append(self.bin_widths[j](y))
            bin_heights.append(self.bin_heights[j](y))
    
        # Define outputs for knot slopes
        for j in range(self.spline_knots-1):
            knot_slopes.append(self.knot_slopes[j](y))
        
        
        # Return output lists
        return bin_widths, bin_heights, knot_slopes


class Cspline(tfb.Bijector):
    """
    Implementation of a Cspline
    """

    def __init__(self, 
                 ndims,
                 rem_dims,
                 spline_knots,
                 range_min,
                 n_hidden = [64, 64, 64],
                 activation = 'relu',
                 use_bias = True,
                 kernel_initializer = 'glorot_uniform', #'he_normal', # 
                 bias_initializer = 'zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None, 
                 activity_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 use_batch_norm = False,
                 forward_min_event_ndims = 1,
                 validate_args: bool = False,
                 name = 'Cspline'):
        
        # Initialize tfb.Bijector superclass
        super(Cspline, self).__init__(validate_args = validate_args,
                                      forward_min_event_ndims = forward_min_event_ndims)
   
        # Check that rem_dims is valid (>1 and <ndims-1)
        if rem_dims<1 or rem_dims>ndims-1:
            raise ValueError("ERROR: rem_dims must be 1<rem_dims<ndims-1")
        
        # Set attributes
        self.range_min = range_min
        self.tran_ndims = ndims-rem_dims
        self.spline_knots = spline_knots
        self.use_batch_norm = use_batch_norm
        
        # Define NN for Cspline bijector
        nn_layer = NN(self.tran_ndims,
                      self.spline_knots,
                      self.range_min,
                      n_hidden,
                      activation,
                      use_bias,
                      kernel_initializer,
                      bias_initializer,
                      kernel_regularizer,
                      bias_regularizer,
                      activity_regularizer,
                      kernel_constraint,
                      bias_constraint,
                      use_batch_norm)
        
        # Define input layer
        x = tf.keras.Input((rem_dims,))
        
        # Define NN output
        bin_widths, bin_heights, knot_slopes = nn_layer(x)
        
        # Define Cspline bijector as tf.keras.Model with input x and output [bin_widths, bin_heights, knot_slopes]
        self.nn = Model(x, [bin_widths, bin_heights, knot_slopes])

    def _bijector_fn(self, x):
        """
        This function implements the _bijection_fn method of the tfb.Bijector superclass needed to define a bijector object.
        It returns the RationalQuadraticSpline bijector with the parameters calculated by the NN.
        """
        def reshape():
            """
            This function reshapes the outputs bin_widths, bin_heights, and knot_slopes from the NN 
            into the correct shape for the Cspline bijector (also transforming them into tf.Tensors).
            """
            # Define output of NN
            print(f"Input to NN x: {x[0:5]}") 
            ii=0        
            for layer in self.nn.layers:
                ii = ii+1
                print("=====================================")
                print(f"Layer {ii}: {layer.name}")
                print("=====================================")
                print(f"type(Layer): {type(layer)}")
                if isinstance(layer, NN):
                    ih=0
                    ibw=0
                    ibh=0
                    iks=0
                    print("============ hidden layers ============")
                    for layer2 in layer.layer_list:
                        if isinstance(layer2, Dense):
                            ih = ih+1
                            print("=====================================")
                            print(f"Layer2 {ih}: {layer2.name}")
                            print(f"type(Layer2): {type(layer2)}")
                            weights = layer2.kernel
                            tf.print(f"weights: {weights}")
                            bias = layer2.bias
                            tf.print(f"bias: {bias}")
                    print("============ bin_widths layers ============")
                    for layer2 in layer.bin_widths:
                        if isinstance(layer2, Dense):
                            ibw = ibw+1
                            print("=====================================")
                            print(f"Layer2 {ibw}: {layer2.name}")
                            print(f"type(Layer2): {type(layer2)}")
                            weights = layer2.kernel
                            tf.print(f"weights: {weights}")
                            bias = layer2.bias
                            tf.print(f"bias: {bias}")
                    print("============ bin_heights layers ============")
                    for layer2 in layer.bin_heights:
                        if isinstance(layer2, Dense):
                            ibh = ibh+1
                            print("=====================================")
                            print(f"Layer2 {ibh}: {layer2.name}")
                            print(f"type(Layer2): {type(layer2)}")
                            weights = layer2.kernel
                            tf.print(f"weights: {weights}")
                            bias = layer2.bias
                            tf.print(f"bias: {bias}")
                    print("============ knot_slopes layers ============")
                    for layer2 in layer.knot_slopes:
                        if isinstance(layer2, Dense):
                            iks = iks+1
                            print("=====================================")
                            print(f"Layer2 {iks}: {layer2.name}")
                            print(f"type(Layer2): {type(layer2)}")
                            weights = layer2.kernel
                            tf.print(f"weights: {weights}")
                            bias = layer2.bias
                            tf.print(f"bias: {bias}")
                            #if tf.math.is_nan(tf.reduce_sum(weights)):
                            #    print("NaN values found in layer weights")
                            #    break
                #if hasattr(layer, 'get_weights'):
                #    weights = layer.get_weights()
                #else:
                #    print(f"Layer {layer.name} does not have weights")
                #for weight in weights:
                #    if np.isnan(weight).any():
                #        print("NaNs found in layer weights")
            #    x = layer(x)
            #    if tf.math.is_nan(tf.reduce_sum(x)):
            #        print("NaN values found after layer:", layer.name)
            #        break
            [bin_widths, bin_heights, knot_slopes] = self.nn(x)
            print(f"Output of NN bin_widths: {bin_widths[0:5]}")
            print(f"Output of NN bin_heights: {bin_heights[0:5]}")
            print(f"Output of NN knot_slopes: {knot_slopes[0:5]}")
            
            # Defning the factor by which the parameters are scaled to get the correct range.
            factor = tf.constant(2*abs(self.range_min), dtype = tf.float32)
            
            # Processing bin_widths
            bin_widths = tf.convert_to_tensor(bin_widths)
            bin_widths = tf.transpose(bin_widths, perm = [1, 2, 0])
            bin_widths = tf.math.softmax(bin_widths)
            bin_widths = tf.math.scalar_mul(factor, bin_widths)
                        
            # Processing bin_heights
            bin_heights = tf.convert_to_tensor(bin_heights)
            bin_heights = tf.transpose(bin_heights, perm=[1, 2, 0])
            bin_heights = tf.math.softmax(bin_heights)
            bin_heights=tf.math.scalar_mul(factor, bin_heights)

            # Processing knot_slopes
            knot_slopes = tf.convert_to_tensor(knot_slopes)
            knot_slopes = tf.transpose(knot_slopes, perm = [1, 2, 0])
            knot_slopes = tf.math.softplus(knot_slopes)
           
            # Return processed outputs
            return bin_widths,bin_heights,knot_slopes
        
        bin_widths, bin_heights, knot_slopes = reshape()
        print(f"Input to RQS bin_widths: {bin_widths}")
        print(f"Input to RQS bin_heights: {bin_heights}")
        print(f"Input to RQS knot_slopes: {knot_slopes}")
        
        # Evaluates the RQS for the given inputs (bin_widths, bin_heights, knot_slopes)
        RQS = RationalQuadraticSpline(bin_widths = bin_widths,
                                      bin_heights = bin_heights,
                                      knot_slopes = knot_slopes,
                                      range_min = self.range_min,
                                      validate_args = False)
        
        print(f"Output of RQS: {RQS[0:5]}")
        # Return RQS
        return RQS
    
    def _forward(self, x):
        """
        Forward transformation of the Cspline bijector.
        """
        print("\nCalls forward method")
        print(f"Input x: {x[0:5]}")
        x_a = x[:, :self.tran_ndims]
        x_b = x[:, self.tran_ndims:]
        y_b = x_b
        y_a = self._bijector_fn(x_b).forward(x_a)
        y = tf.concat([y_a, y_b], axis = -1)
        print(f"Output y: {y[0:5]}")
        return y
    
    def _inverse(self, y):
        """
        Inverse transformation of the Cspline bijector.
        """
        print("\nCalls inverse method")
        print(f"Input y: {y[0:5]}")
        y_a = y[:, :self.tran_ndims]
        y_b = y[:, self.tran_ndims:]
        x_b = y_b
        x_a = self._bijector_fn(y_b).inverse(y_a)
        x = tf.concat([x_a, x_b], axis = -1)
        print(f"Output x: {x[0:5]}")
        return x
    
    def _forward_log_det_jacobian(self, x):
        """
        Forward log det jacobian of the Cspline bijector.
        """
        print("\nCalls forward_log_det_jacobian method")
        print(f"Input x: {x[0:5]}")
        x_a = x[:, :self.tran_ndims]
        x_b = x[:, self.tran_ndims:]
        fldj = self._bijector_fn(x_b).forward_log_det_jacobian(x_a, event_ndims = 1)
        print(f"Output fldj: {fldj[0:5]}")
        return fldj
    
    def _inverse_log_det_jacobian(self, y):
        """
        Inverse log det jacobian of the Cspline bijector.
        """
        print("\nCalls inverse_log_det_jacobian method")
        print(f"Input y: {y[0:5]}")
        y_a = y[:, :self.tran_ndims]
        y_b = y[:, self.tran_ndims:]
        iljd = self._bijector_fn(y_b).inverse_log_det_jacobian(y_a, event_ndims = 1)
        print(f"Output iljd: {iljd[0:5]}")
        return iljd