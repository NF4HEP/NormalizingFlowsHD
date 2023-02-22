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
    Neural Network Architecture for calcualting s and t for Real-NVP
  
    """
    def __init__(self, tran_dims, spline_knots,range_min, n_hidden=[5,5],activation="relu",use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None):
        super(NN, self).__init__()
        
        
        #print(n_hidden)
        self.tran_dims=tran_dims
        self.range_min=range_min
        self.spline_knots=spline_knots
        layer_list = []
        for i, hidden in enumerate(n_hidden):
            layer_list.append(Dense(hidden, activation=activation,use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint))
        self.layer_list = layer_list
       # self.log_s_layer = Dense(input_shape, activation="tanh", name='log_s')
       # self.t_layer = Dense(input_shape, name='t')
        
        bin_widths=[]
        bin_heights=[]
        knot_slopes=[]
      
        for _ in range(spline_knots):
            bin_widths.append(Dense(tran_dims,activation=activation))
            bin_heights.append(Dense(tran_dims,activation=activation))
    
        for _ in range(spline_knots-1):
            knot_slopes.append(Dense(tran_dims,activation=activation))
            
        
        '''
        bin_widths=Dense(self.tran_dims*spline_knots,activation="softmax")
        bin_heights=Dense(self.tran_dims*spline_knots,activation="softmax")
        knot_slopes=Dense(self.tran_dims*(spline_knots-1),activation="softplus")
        '''
        
        
        
        self.bin_widths=bin_widths
        self.bin_heights=bin_heights
        self.knot_slopes=knot_slopes
        
       
        

    def call(self, x):
      
        y = x
      
        for layer in self.layer_list:
            y = layer(y)
            
        
        bin_widths=[]
        bin_heights=[]
        knot_slopes=[]
        for j in range(self.spline_knots):
            bin_widths.append(self.bin_widths[j](y))
            #print('self bin)widths')
            #print(self.bin_widths[j](y))
            bin_heights.append(self.bin_heights[j](y))
    
        for j in range(self.spline_knots-1):
            knot_slopes.append(self.knot_slopes[j](y))
        '''
        #print('ageqrgkwegbkehjb kehbgqkeruhbgke')
        #print(y)
        
        #print('hhjhjss')
        bin_widths=tf.keras.layers.Reshape((self.tran_dims,spline_knots))(bin_widths)
        #print('heeeeeyy')
        bin_heights=tf.keras.layers.Reshape((self.tran_dims,spline_knots))(bin_heights)
        #print('crrroooor')
        knot_slopes=tf.keras.layers.Reshape((self.tran_dims,spline_knots-1))(knot_slopes)
        #print('ggooog')
        
        bin_widths = Lambda(lambda x: x *(10+abs(self.range_min)))(bin_widths)
        bin_heights = Lambda(lambda x: x *(10+abs(self.range_min)))(bin_heights)
        knot_slopes = Lambda(lambda x: x +10e-5)(knot_slopes)
        '''
        '''
        
        bin_widths=self.bin_widths(y)
        bin_heights=self.bin_heights(y)
        knot_slopes=self.knot_slopes(y)
        '''
        
        return bin_widths,bin_heights,knot_slopes




class Cspline(tfb.Bijector):
    """
    Implementation of a Cspline

    """

    def __init__(self, ndims,rem_dims,spline_knots,range_min,n_hidden=[64,64,64],activation='relu',use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, forward_min_event_ndims=1, validate_args: bool = False,name='Cspline'):
        super(Cspline, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims
        )
   
        
        if rem_dims<1 or rem_dims>ndims-1:
            print('ERROR: rem_dims must be 1<rem_dims<ndims-1')
            exit()
       
       
        self.range_min=range_min
        self.tran_ndims=ndims-rem_dims
   
        self.spline_knots=spline_knots
        #input_shape = input_shape // 2
        nn_layer = NN(self.tran_ndims,self.spline_knots,self.range_min, n_hidden,activation,use_bias,
    kernel_initializer,
    bias_initializer, kernel_regularizer,
    bias_regularizer, activity_regularizer, kernel_constraint,
    bias_constraint)
        x = tf.keras.Input((rem_dims,))
        bin_widths,bin_heights,knot_slopes = nn_layer(x)
        
        self.nn = Model(x, [bin_widths,bin_heights,knot_slopes])

    
    
 
    #@tf.function
    def _bijector_fn(self, x):
    
        
        def reshape():
        
        
            [bin_widths,bin_heights,knot_slopes]=self.nn(x)
            
            #print('hello')
           
            #print(tf.shape(x)[0])
            #print(output[0])
            
            #output=tf.reshape(output, (x.shape[0],self.tran_ndims,3*spline_knots-1), name=None)
            #print('bin_widths')
            
            #print(bin_widths)
            #bin_widths=output[:,:,:spline_knots]
            #bin_widths=tf.reshape(output[0], (x.shape[0],self.tran_ndims,self.spline_knots), name=None)
            bin_widths=tf.convert_to_tensor(bin_widths)
            
            
            #print('to tensor bin widhts')
            #print(tf.shape(bin_widths))
            bin_widths=tf.transpose(bin_widths,perm=[1,2,0])
            #print(tf.shape(bin_widths))
            #print(bin_widths)
            #print('transposed bin widhts')
            #print(bin_widhts_transp)
            #bin_widths=tf.reshape(bin_widths, (tf.shape(x)[0],self.tran_ndims,self.spline_knots), name=None)
            bin_widths=tf.math.softmax(bin_widths)
            #print(bin_widths)
            bin_widths=tf.math.scalar_mul(tf.constant(2*abs(self.range_min),dtype=tf.float32),bin_widths)
                        
            #print('reshaped bin_widths')
            #print(bin_widths)
            
            #print('reshape bin heights')
            #print(tf.shape(bin_heights))
            #bin_heights=tf.reshape(output[1], (x.shape[0],self.tran_ndims,self.spline_knots), name=None)
            bin_heights=tf.convert_to_tensor(bin_heights)
            bin_heights=tf.transpose(bin_heights,perm=[1,2,0])
            bin_heights=tf.math.softmax(bin_heights)
            
            #bin_heights=tf.reshape(bin_heights, (tf.shape(x)[0],self.tran_ndims,self.spline_knots), name=None)
            bin_heights=tf.math.scalar_mul(tf.constant(2*abs(self.range_min),dtype=tf.float32),bin_heights)
            #print('bin_heights')
            #print(tf.shape(bin_heights))
            
            #knot_slopes=tf.reshape(output[2], (x.shape[0],self.tran_ndims,self.spline_knots-1), name=None)+tf.constant(1e-5,dtype=tf.float32)
            #print('knot slopes')
            #print(tf.shape(knot_slopes))
            knot_slopes=tf.convert_to_tensor(knot_slopes)
            knot_slopes=tf.transpose(knot_slopes,perm=[1,2,0])
            knot_slopes=tf.math.softplus(knot_slopes)

            #print('reshape knot slopes')
            
            #print('knot_slopes')
            #print(tf.shape(knot_slopes))
            #knot_slopes=tf.math.scalar_mul(2*abs(self.range_min),knot_slopes)
           
            return bin_widths,bin_heights,knot_slopes
        
        
        
       
        bin_widths,bin_heights,knot_slopes=reshape()
        #print('hey')
        #RQS=tf.cast(RQS,dtype=tf.float32)
        return RationalQuadraticSpline(
    bin_widths=bin_widths, bin_heights=bin_heights, knot_slopes=knot_slopes, range_min=self.range_min, validate_args=False)
    
    def _forward(self, x):
        #x_a, x_b = tf.split(x, 2, axis=-1)

        x_a=x[:,:self.tran_ndims]
        x_b=x[:,self.tran_ndims:]
        #print(x_b)
        y_b = x_b
        #print('gggggghhhggg')
        y_a = self._bijector_fn(x_b).forward(x_a)
        #print('did i get here?')
        #print('y_a')
        #print(y_a)
        #print(y_a)
        y = tf.concat([y_a, y_b], axis=-1)
        
        return y
    
    def _inverse(self, y):

        #y_a, y_b = tf.split(y, 2, axis=-1)
        #print('niverse')
        y_a=y[:,:self.tran_ndims]
        y_b=y[:,self.tran_ndims:]
        x_b = y_b
        x_a = self._bijector_fn(y_b).inverse(y_a)
        x = tf.concat([x_a, x_b], axis=-1)
        #print('hello')
        return x
    
    def _forward_log_det_jacobian(self, x):
        #x_a, x_b = tf.split(x, 2, axis=-1)
        x_a=x[:,:self.tran_ndims]
        x_b=x[:,self.tran_ndims:]
    
        return self._bijector_fn(x_b).forward_log_det_jacobian(x_a, event_ndims=1)
    
    def _inverse_log_det_jacobian(self, y):
        #y_a, y_b = tf.split(y, 2, axis=-1)
        y_a=y[:,:self.tran_ndims]
        y_b=y[:,self.tran_ndims:]
        return self._bijector_fn(y_b).inverse_log_det_jacobian(y_a, event_ndims=1)
