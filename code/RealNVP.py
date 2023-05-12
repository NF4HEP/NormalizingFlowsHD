

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:36:34 2019

@author: reyes-gonzalez
"""
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

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

from tensorflow.keras import Model, Input,Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization,Add,Dense,ReLU,Layer, Conv2D, Reshape
from tensorflow.keras.optimizers import Adam
from statistics import mean,median


tfd = tfp.distributions
tfb = tfp.bijectors

#import Distributions,Metrics,Trainer

def get_conv_resnet(input_shape, filters):
    """
    This function should build a CNN ResNet model according to the above specification,
    using the functional API. The function takes input_shape as an argument, which should be
    used to specify the shape in the Input layer, as well as a filters argument, which
    should be used to specify the number of filters in (some of) the convolutional layers.
    Your function should return the model.
    """

    h0=Input(shape=input_shape)
    
    x = Conv2D(kernel_size=3,filters=filters,padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(5e-5),activation="relu")(h0)
    x = BatchNormalization()(x)
    x = Conv2D(kernel_size=3,filters=input_shape[-1],padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(5e-5),activation="relu")(x)
    x = BatchNormalization()(x)
    h1= Add()([h0,x])
    y = Conv2D(kernel_size=3,filters=filters,padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(5e-5),activation="relu")(h1)
    y = BatchNormalization()(y)
    y = Conv2D(kernel_size=3,filters=input_shape[-1],padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(5e-5),activation="relu")(y)
    y = BatchNormalization()(y)
    y = Add()([h1,y])
    h2 = Conv2D(kernel_size=3,filters=2*input_shape[-1],padding='SAME', kernel_regularizer=tf.keras.regularizers.l2(5e-5),activation="linear")(y)
    
    h2= tf.split(h2,num_or_size_splits=2, axis=-1)
    
    outputs = [h2[0],tf.keras.activations.tanh(h2[1])]
    
    model= Model(inputs=h0, outputs= outputs)
    
   
    
    
    
    return model
    
    
def get_nn(ndims):
    """
    This function should build a CNN ResNet model according to the above specification,
    using the functional API. The function takes input_shape as an argument, which should be
    used to specify the shape in the Input layer, as well as a filters argument, which
    should be used to specify the number of filters in (some of) the convolutional layers.
    Your function should return the model.
    """


   
    
    h0=Input(shape=(ndims,))
    
    y = Dense(128,activation="relu")(h0)
    #y = BatchNormalization()(y)
    #h1= Add()([h0,x])
    #y = Dense(128,activation="relu")(y)
    y = Dense(128,activation="relu")(y)
    #y = BatchNormalization()(y)
    y = Dense(128,activation="relu")(y)
   # y = BatchNormalization()(y)
    #y = Add()([h1,y])
    #y = Dense(ndims*2,activation="relu")(y)
    
    shift=Dense(ndims,activation="relu")(y)
    log_scale=Dense(ndims,activation="tanh")(y)
    
    #h2= tf.split(y,num_or_size_splits=2, axis=-1)
    
    #outputs = [h2[0],tf.keras.activations.tanh(h2[1])]
    outputs = [shift,log_scale]
    
    model= Model(inputs=h0, outputs= outputs)

   
   
    
    
    return model
    
######## BINARY MASKS ################


class NN(Layer):
    """
    Neural Network Architecture for calcualting s and t for Real-NVP
  
    """
    def __init__(self, input_shape, n_hidden=[128,128,128], activation="relu",use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None):
        super(NN, self).__init__()
        layer_list = []
        for i, hidden in enumerate(n_hidden):
            layer_list.append(Dense(hidden, activation=activation))
        self.layer_list = layer_list
        self.log_s_layer = Dense(input_shape, activation="tanh", name='log_s')
        self.t_layer = Dense(input_shape, name='t')

    def call(self, x):
        y = x
        for layer in self.layer_list:
            y = layer(y)
        log_s = self.log_s_layer(y)
        t = self.t_layer(y)
        return t, log_s
    
class RealNVP(tfb.Bijector):
    """
    Implementation of a Real-NVP for Denisty Estimation. L. Dinh “Density estimation using Real NVP,” 2016.

    """

    def __init__(self, ndims,rem_dims, n_hidden=[24,24],activation='relu', forward_min_event_ndims=1,use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, validate_args: bool = False):
        super(RealNVP, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims
        )
       
        if rem_dims<1 or rem_dims>ndims-1:
            print('ERROR: rem_dims must be 1<rem_dims<ndims-1')
            exit()
        
        self.tran_ndims=ndims-rem_dims
        #input_shape = input_shape // 2
        nn_layer = NN(self.tran_ndims, n_hidden,activation,use_bias,
    kernel_initializer,
    bias_initializer, kernel_regularizer,
    bias_regularizer, activity_regularizer, kernel_constraint,
    bias_constraint)
        x = tf.keras.Input((rem_dims,))
        t,log_s = nn_layer(x)
        self.nn = Model(x, [t, log_s])


    
    def _bijector_fn(self, x):
        t, log_s = self.nn(x)
        #print('this is t')
        #print(t)
        #print(tfb.affine_scalar.AffineScalar(shift=t, log_scale=log_s))
        
        
        affine_scalar=tfb.Chain([tfb.Shift(t),tfb.Scale(log_scale=log_s)])
        
        return affine_scalar

    def _forward(self, x):
        #x_a, x_b = tf.split(x, 2, axis=-1)
        x_a=x[:,:self.tran_ndims]
        x_b=x[:,self.tran_ndims:]
        #print('x_a')
        #print(x_a)
   
        y_b = x_b
        y_a = self._bijector_fn(x_b).forward(x_a)
        y = tf.concat([y_a, y_b], axis=-1)
        
        return y

    def _inverse(self, y):
        #y_a, y_b = tf.split(y, 2, axis=-1)
        y_a=y[:,:self.tran_ndims]
        y_b=y[:,self.tran_ndims:]
        x_b = y_b
        x_a = self._bijector_fn(y_b).inverse(y_a)
        x = tf.concat([x_a, x_b], axis=-1)
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

'''

ndims=9
nsamples=100
n_epochs=3
base_dist=Distributions.gaussians(ndims)
    
targ_dist=Distributions.mixture_of_gaussians(ndims)
X_data=targ_dist.sample(nsamples)
#trainable_distribution
input_shape=X_data.shape
rem_dims=9
bijector=RealNVP(ndims,rem_dims)
nf_dist=tfd.TransformedDistribution(base_dist,bijector)
sample=nf_dist.sample(nsamples)
history=Trainer.train_dist_routine(ndims,nf_dist, X_data,n_epochs=n_epochs, batch_size=40, n_disp=2)
print(sample)
'''
'''

bijectors=[]
permutation=tf.cast(np.concatenate((np.arange(int(ndims/2),ndims),np.arange(0,int(ndims/2)))), tf.int32)
for i in range(5):
    #bijectors.append(tfb.BatchNormalization())
    bijectors.append(RealNVP(ndims))
    bijectors.append(tfp.bijectors.Permute(permutation))
bijector = tfb.Chain(bijectors=list(reversed(bijectors[:-1])), name='chain_of_real_nvp')
nf_dist=tfd.TransformedDistribution(base_dist,bijector)
#t_losses,v_losses=Trainer.custom_train_routine(nf_dist,targ_dist,nsamples,n_epochs)
history=Trainer.train_dist_routine(ndims,nf_dist, X_data,n_epochs=n_epochs, batch_size=4096, n_disp=10)
print(nf_dist.sample(11))
print(nf_dist.log_prob(nf_dist.sample(11)))
print('compare base to nf')
base_sample=base_dist.sample(10)

print(base_sample)
print(nf_dist.bijector.forward(base_sample))

kl_divergence=Metrics.KL_divergence(targ_dist,nf_dist,nsamples=10000)
print('kl_divergence')
print(kl_divergence)

ks_test_list=Metrics.KS_test(targ_dist,nf_dist,base_dist,ndims,nsamples=1000)
ks_test_mean=mean(ks_test_list)
ks_test_median=median(ks_test_list)
print('ks_test')
print(ks_test_list)
'''
